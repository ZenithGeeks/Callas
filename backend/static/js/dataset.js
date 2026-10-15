/* /dataset review page: per-turn playback, whole-call playback that highlights the current
 * transcript row, keyboard review (J/K move, A approve, R reject, Space play), and moving on
 * to the next call after a decision with an undo toast.
 * Listeners sit on document, so they keep working after htmx swaps in another call. */
(function () {
  "use strict";

  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ---- helpers --------------------------------------------------------------------------
  function formatTime(seconds) {
    var s = Math.max(0, Math.floor(seconds || 0));
    return Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0");
  }

  function setPressed(btn, on) {
    if (!btn) return;
    btn.setAttribute("aria-pressed", on ? "true" : "false");
    btn.dataset.playing = on ? "true" : "false";
  }

  function queueRows() {
    return Array.prototype.slice.call(document.querySelectorAll("[data-row]"));
  }

  // Marks one queue row as the open call (by id); every other row loses the mark.
  function markCurrentRow(id) {
    queueRows().forEach(function (r) {
      if (r.dataset.row === id) r.setAttribute("aria-current", "true");
      else r.removeAttribute("aria-current");
    });
  }

  // The id of the call shown in the right column, or "".
  function openCallId() {
    var c = document.querySelector("#dataset-call [data-call]");
    return c ? c.dataset.call : "";
  }

  // ---- one turn's audio -----------------------------------------------------------------
  var turnAudio = new Audio();
  var turnBtn = null; // the mini play button whose turn is loaded

  function playTurn(btn) {
    stopCall();
    if (turnBtn === btn && !turnAudio.paused) return stopTurn();
    stopTurn();
    turnBtn = btn;
    turnAudio.src = btn.dataset.playSrc;
    setPressed(btn, true);
    // play() rejects when the file is gone or the browser blocks it; undo the pressed state.
    turnAudio.play().catch(function () {
      if (turnBtn === btn) stopTurn();
    });
  }

  function stopTurn() {
    turnAudio.pause();
    setPressed(turnBtn, false);
    turnBtn = null;
  }

  turnAudio.addEventListener("ended", stopTurn);

  // ---- whole-call audio -----------------------------------------------------------------
  var speakingRow = null; // transcript row highlighted for the current playback time
  var boundAudio = new WeakSet(); // not a data attribute: a restored history page keeps attributes but not listeners

  function callAudio() {
    return document.querySelector("[data-call-audio]");
  }

  function stopCall() {
    var a = callAudio();
    if (a && !a.paused) a.pause();
  }

  function rowAtTime(t) {
    var rows = document.querySelectorAll("#dtranscript [data-start]");
    for (var i = 0; i < rows.length; i++) {
      var r = rows[i];
      if (t >= parseFloat(r.dataset.start) && t < parseFloat(r.dataset.end)) return r;
    }
    return null;
  }

  // Highlights the row being spoken, scrolls it into view if it is off screen, updates the clock.
  function followPlayback(a) {
    var row = a.paused && a.currentTime === 0 ? null : rowAtTime(a.currentTime);
    if (row !== speakingRow) {
      if (speakingRow) speakingRow.removeAttribute("data-speaking");
      if (row) {
        row.setAttribute("data-speaking", "");
        var box = row.getBoundingClientRect();
        if (box.top < 0 || box.bottom > window.innerHeight) {
          row.scrollIntoView({ block: "center", behavior: reduceMotion ? "auto" : "smooth" });
        }
      }
      speakingRow = row;
    }
    var clock = document.querySelector("[data-call-time]");
    if (clock) clock.textContent = formatTime(a.currentTime);
  }

  function clearSpeakingRow() {
    if (speakingRow) speakingRow.removeAttribute("data-speaking");
    speakingRow = null;
  }

  // Swaps the "Play call" / "Pause" labels on the call's play button.
  function showCallPlaying(on) {
    var btn = document.querySelector("[data-call-play]");
    if (!btn) return;
    setPressed(btn, on);
    btn.querySelector("[data-when-paused]").hidden = on;
    btn.querySelector("[data-when-playing]").hidden = !on;
  }

  // Runs for each newly swapped-in or restored call.
  function bindCallAudio() {
    var a = callAudio();
    clearSpeakingRow();
    document.querySelectorAll("[data-speaking]").forEach(function (r) {
      r.removeAttribute("data-speaking");
    });
    if (!a || boundAudio.has(a)) return;
    boundAudio.add(a);
    a.addEventListener("timeupdate", function () { followPlayback(a); });
    a.addEventListener("play", function () { showCallPlaying(true); });
    a.addEventListener("pause", function () { showCallPlaying(false); });
    a.addEventListener("ended", function () {
      showCallPlaying(false);
      clearSpeakingRow();
    });
  }

  function playCall(a) {
    a.play().catch(function () { showCallPlaying(false); });
  }

  // With seekTo (seconds): play from there. Without: toggle play and pause.
  function toggleCall(seekTo) {
    var a = callAudio();
    if (!a) return;
    stopTurn();
    if (seekTo != null) {
      a.currentTime = seekTo;
      playCall(a);
    } else if (a.paused) {
      playCall(a);
    } else {
      a.pause();
    }
  }

  // ---- clicks: play buttons, seek times, queue rows -------------------------------------
  document.addEventListener("click", function (e) {
    var play = e.target.closest("[data-play-src]");
    if (play) return playTurn(play);
    if (e.target.closest("[data-call-play]")) return toggleCall(null);
    var seek = e.target.closest("[data-seek]");
    if (seek) return toggleCall(parseFloat(seek.dataset.seek));
    var row = e.target.closest("[data-row]");
    if (row) markCurrentRow(row.dataset.row);
  });

  // ---- keyboard -------------------------------------------------------------------------
  // Opens the next (step 1) or previous (step -1) row in the queue.
  function moveSelection(step) {
    var rows = queueRows();
    if (!rows.length) return;
    var i = rows.findIndex(function (r) { return r.hasAttribute("aria-current"); });
    var next = rows[Math.min(rows.length - 1, Math.max(0, i + step))];
    if (next && next !== rows[i]) {
      next.click();
      next.scrollIntoView({ block: "nearest" });
    }
  }

  // Keys are matched by position (e.code), so they work on a Thai keyboard layout too.
  document.addEventListener("keydown", function (e) {
    // Keys pressed with nothing focused come from the document, which has no closest().
    var target = e.target && e.target.closest ? e.target : document.body;
    // Enter in the note would submit the form through its first button, which is Approve.
    if (e.key === "Enter" && target.id === "note") {
      e.preventDefault();
      return;
    }
    if (e.repeat || e.ctrlKey || e.metaKey || e.altKey) return;
    if (target.closest("input, textarea, select, [contenteditable]")) return;
    var code = e.code;
    if (code === "KeyJ") moveSelection(1);
    else if (code === "KeyK") moveSelection(-1);
    else if (code === "KeyA" || code === "KeyR") {
      var btn = document.querySelector(
        '#review-bar [data-key="' + code.slice(3).toLowerCase() + '"]'
      );
      if (btn) btn.click();
    } else if (code === "Space" && !target.closest("button, a, summary")) {
      e.preventDefault();
      toggleCall(null);
    }
  });

  // ---- auto-advance setting -------------------------------------------------------------
  var AUTO_NEXT_KEY = "callas.dataset.autoNext"; // "on" or "off"

  function autoNextEnabled() {
    try {
      return localStorage.getItem(AUTO_NEXT_KEY) !== "off";
    } catch (err) {
      return true;
    }
  }

  // The checkbox is part of each swapped-in decision bar, so it is reset from storage every time.
  function syncAutoNextBox() {
    var box = document.querySelector("[data-auto-next]");
    if (box) box.checked = autoNextEnabled();
  }

  document.addEventListener("change", function (e) {
    if (!e.target.matches("[data-auto-next]")) return;
    try {
      localStorage.setItem(AUTO_NEXT_KEY, e.target.checked ? "on" : "off");
    } catch (err) {
      /* private mode: the box still works for this page */
    }
  });

  // ---- after a decision: announcement, toast, then the next call ------------------------
  var nextRowId = ""; // the row after the open one, noted before the request moves or removes it
  var undoId = ""; // the call whose Undo request is in flight
  var toastTimer = null;
  var hideTimer = null;

  function announce(text) {
    var live = document.getElementById("review-live");
    if (!live) return;
    live.textContent = "";
    // Clearing first makes a repeated message read out again.
    setTimeout(function () { live.textContent = text; }, 50);
  }

  function showToast(text, id) {
    var toast = document.getElementById("toast");
    if (!toast) return;
    toast.querySelector("[data-toast-text]").textContent = text;
    toast.dataset.id = id;
    toast.hidden = false;
    toast.classList.remove("is-out");
    clearTimeout(toastTimer);
    clearTimeout(hideTimer); // the previous toast's hide must not close this one
    toastTimer = setTimeout(function () {
      toast.classList.add("is-out");
      hideTimer = setTimeout(function () { toast.hidden = true; }, 250); // lets the fade-out finish
    }, 6000);
  }

  function hideToast() {
    var toast = document.getElementById("toast");
    clearTimeout(toastTimer);
    clearTimeout(hideTimer);
    if (toast) toast.hidden = true;
  }

  // Loads a call into the right column the same way a queue click does; htmx pushes the URL.
  function openCall(id) {
    var q = encodeURIComponent(id);
    htmx.ajax("GET", "/dataset/call/" + q, {
      target: "#dataset-call",
      swap: "innerHTML show:#dataset-call:top",
      push: "/dataset?id=" + q,
    });
    markCurrentRow(id);
  }

  function isReviewRequest(detail) {
    return /\/dataset\/call\/[^/]+\/review$/.test((detail.pathInfo || {}).requestPath || "");
  }

  // The form is replaced by the swap, so htmx reports afterRequest on an ancestor; the form
  // itself is still in requestConfig.
  function isReviewForm(detail) {
    var src = detail.requestConfig && detail.requestConfig.elt;
    return !!(src && src.matches && src.matches("[data-review-form]"));
  }

  // Undo: set the call back to pending (the server keeps its note), then reopen it. The toast
  // stays until the request succeeds.
  document.addEventListener("click", function (e) {
    var btn = e.target.closest("[data-toast-undo]");
    if (!btn) return;
    var toast = document.getElementById("toast");
    var type = document.getElementById("f-type");
    var view = document.getElementById("f-view");
    undoId = toast.dataset.id;
    btn.disabled = true;
    htmx.ajax("POST", "/dataset/call/" + encodeURIComponent(undoId) + "/review", {
      values: {
        status: "pending",
        keep_note: "1",
        type: type ? type.value : "",
        view: view ? view.value : "",
      },
      swap: "none",
    });
  });

  function finishUndo(ok) {
    var id = undoId;
    undoId = "";
    var btn = document.querySelector("[data-toast-undo]");
    if (btn) btn.disabled = false;
    if (!ok) {
      showToast("Could not undo. Try again.", id);
      return;
    }
    hideToast();
    announce("Review cleared");
    // The call may have left the list when it was decided; a fresh queue puts it back.
    var type = document.getElementById("f-type");
    var view = document.getElementById("f-view");
    htmx
      .ajax("GET", "/dataset", {
        target: "#queue",
        swap: "innerHTML",
        values: { type: type ? type.value : "", view: view ? view.value : "" },
      })
      .then(function () { openCall(id); });
  }

  function finishReview(ok) {
    var next = nextRowId;
    nextRowId = "";
    if (!ok) {
      announce("Could not save the review. Try again.");
      return;
    }
    var bar = document.getElementById("review-bar");
    if (!bar || bar.dataset.review === "pending") return; // Undo from the bar: stay on this call
    var word = bar.dataset.review === "approved" ? "Approved" : "Rejected";
    announce(word + " " + bar.dataset.name);
    if (!autoNextEnabled()) return;
    showToast(word + " " + bar.dataset.name, bar.dataset.id);
    var row = next && document.querySelector('[data-row="' + next + '"]');
    if (row) {
      row.click();
      row.scrollIntoView({ block: "nearest" });
    }
  }

  document.addEventListener("htmx:beforeRequest", function (e) {
    if (!isReviewRequest(e.detail) || !isReviewForm(e.detail)) return;
    var bar = document.getElementById("review-bar");
    var rows = queueRows();
    var i = rows.findIndex(function (r) { return bar && r.dataset.row === bar.dataset.id; });
    nextRowId = rows[i + 1] ? rows[i + 1].dataset.row : "";
  });

  document.addEventListener("htmx:afterRequest", function (e) {
    if (!isReviewRequest(e.detail)) return;
    if (isReviewForm(e.detail)) finishReview(e.detail.successful);
    else if (undoId) finishUndo(e.detail.successful);
  });

  // A request that got no answer at all: clear what it left behind.
  function requestFailed(e) {
    if (!isReviewRequest(e.detail)) return;
    if (isReviewForm(e.detail)) finishReview(false);
    else if (undoId) finishUndo(false);
  }
  document.addEventListener("htmx:sendError", requestFailed);
  document.addEventListener("htmx:timeout", requestFailed);

  document.addEventListener("htmx:afterSettle", function (e) {
    syncAutoNextBox();
    // A new call: keyboard and screen reader users land on its heading.
    if (e.detail.target && e.detail.target.id === "dataset-call") {
      var h = document.getElementById("call-h");
      if (h) h.focus({ preventScroll: true });
    }
  });

  // ---- htmx swaps and page load ---------------------------------------------------------
  document.addEventListener("htmx:beforeSwap", function (e) {
    if (e.detail.target && e.detail.target.id === "dataset-call") {
      stopTurn();
      stopCall();
    }
  });
  document.addEventListener("htmx:afterSwap", function (e) {
    bindCallAudio();
    // A new queue (filter or tab) has no current row until it is told which call is open, and
    // a newly opened call moves the mark (after Undo the queue swaps before the call does).
    var id = e.detail.target && e.detail.target.id;
    if (id === "queue" || id === "dataset-call") markCurrentRow(openCallId());
  });
  // Leaving a page must not leave its audio playing.
  document.addEventListener("htmx:beforeHistorySave", function () {
    stopTurn();
    stopCall();
  });
  document.addEventListener("htmx:historyRestore", function () {
    bindCallAudio();
    syncAutoNextBox();
    markCurrentRow(openCallId());
  });
  document.addEventListener("DOMContentLoaded", function () {
    bindCallAudio();
    syncAutoNextBox();
  });
})();
