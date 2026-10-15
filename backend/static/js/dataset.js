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

  // ---- one turn's audio -----------------------------------------------------------------
  var turnAudio = new Audio();
  var turnBtn = null; // the mini play button whose turn is loaded

  function playTurn(btn) {
    stopCall();
    if (turnBtn === btn && !turnAudio.paused) return stopTurn();
    stopTurn();
    turnBtn = btn;
    turnAudio.src = btn.dataset.playSrc;
    turnAudio.play();
    setPressed(btn, true);
  }

  function stopTurn() {
    turnAudio.pause();
    setPressed(turnBtn, false);
    turnBtn = null;
  }

  turnAudio.addEventListener("ended", stopTurn);

  // ---- whole-call audio -----------------------------------------------------------------
  var speakingRow = null; // transcript row highlighted for the current playback time

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

  // Runs for each newly swapped-in call; the dataset flag stops a second binding.
  function bindCallAudio() {
    var a = callAudio();
    speakingRow = null;
    if (!a || a.dataset.bound) return;
    a.dataset.bound = "1";
    a.addEventListener("timeupdate", function () { followPlayback(a); });
    a.addEventListener("play", function () { showCallPlaying(true); });
    a.addEventListener("pause", function () { showCallPlaying(false); });
    a.addEventListener("ended", function () {
      showCallPlaying(false);
      clearSpeakingRow();
    });
  }

  // With seekTo (seconds): play from there. Without: toggle play and pause.
  function toggleCall(seekTo) {
    var a = callAudio();
    if (!a) return;
    stopTurn();
    if (seekTo != null) {
      a.currentTime = seekTo;
      a.play();
    } else if (a.paused) {
      a.play();
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

  document.addEventListener("keydown", function (e) {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    var target = e.target;
    if (target.closest("input, textarea, select, [contenteditable]")) return;
    var key = e.key.toLowerCase();
    if (key === "j") moveSelection(1);
    else if (key === "k") moveSelection(-1);
    else if (key === "a" || key === "r") {
      var btn = document.querySelector('#review-bar [data-key="' + key + '"]');
      if (btn) btn.click();
    } else if (key === " " && !target.closest("button, a, summary")) {
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

  // ---- after a decision: toast, then the next call --------------------------------------
  var reviewSubmitted = false; // set when the decision form is sent, read once the bar is swapped
  var toastTimer = null;

  function showToast(text, id) {
    var toast = document.getElementById("toast");
    if (!toast) return;
    toast.querySelector("[data-toast-text]").textContent = text;
    toast.dataset.id = id;
    toast.hidden = false;
    toast.classList.remove("is-out");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () {
      toast.classList.add("is-out");
      setTimeout(function () { toast.hidden = true; }, 250); // lets the fade-out finish
    }, 6000);
  }

  // Loads a call into the right column the same way a queue click does.
  function openCall(id) {
    htmx.ajax("GET", "/dataset/call/" + id, { target: "#dataset-call", swap: "innerHTML" });
    history.pushState({}, "", "/dataset?id=" + encodeURIComponent(id));
    markCurrentRow(id);
  }

  // Undo: set the call back to pending, then reopen it.
  document.addEventListener("click", function (e) {
    if (!e.target.closest("[data-toast-undo]")) return;
    var toast = document.getElementById("toast");
    var id = toast.dataset.id;
    var type = document.getElementById("f-type");
    var view = document.getElementById("f-view");
    toast.hidden = true;
    htmx
      .ajax("POST", "/dataset/call/" + id + "/review", {
        values: { status: "pending", type: type ? type.value : "", view: view ? view.value : "" },
        swap: "none",
      })
      .then(function () { openCall(id); });
  });

  document.addEventListener("htmx:beforeRequest", function (e) {
    if (e.detail.elt.matches && e.detail.elt.matches("[data-review-form]")) reviewSubmitted = true;
  });

  document.addEventListener("htmx:afterSettle", function () {
    syncAutoNextBox();
    if (!reviewSubmitted) return;
    reviewSubmitted = false;
    var bar = document.getElementById("review-bar");
    if (!bar || bar.dataset.review === "pending") return; // Undo: stay on this call
    if (!autoNextEnabled()) return;
    var word = bar.dataset.review === "approved" ? "Approved" : "Rejected";
    showToast(word + " " + bar.dataset.name, bar.dataset.id);
    moveSelection(1);
  });

  // ---- htmx swaps and page load ---------------------------------------------------------
  document.addEventListener("htmx:beforeSwap", function (e) {
    if (e.detail.target && e.detail.target.id === "dataset-call") {
      stopTurn();
      stopCall();
    }
  });
  document.addEventListener("htmx:afterSwap", bindCallAudio);
  document.addEventListener("DOMContentLoaded", function () {
    bindCallAudio();
    syncAutoNextBox();
  });
})();
