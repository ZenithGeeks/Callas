/* /studio: show a call as the server writes it.
 * The server sends server-sent events whose data is a piece of HTML: "thinking", "plan", "turn"
 * and "verdict" are list items for the thread (an item with the id of an earlier one replaces it,
 * which is how a "thinking" bubble becomes the message), "failed" is an error message and "done"
 * ends the run. A plain EventSource is used instead of the htmx sse extension because an event
 * has to replace one element by id. Closing the tab does not stop the run; the server saves it.
 * The two batch buttons send one call after another through the same form: all six kinds, or
 * a real dataset (100 calls cycling through the scenarios of the chosen kind, stopped when under
 * half pass). When a real-dataset batch ends, for any reason, its token usage is shown above the
 * run (GET /studio/tokens with every call it started). */
(function () {
  "use strict";

  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var source = null;
  var batch = null;
  var REAL_BATCH_SIZE = 100; // calls in one "Generate a real dataset" batch

  function toElement(html) {
    var t = document.createElement("template");
    t.innerHTML = html.trim();
    return t.content.firstElementChild;
  }

  // Put an item in the thread: replace the one with the same id, or add it at the end.
  function put(thread, html) {
    var el = toElement(html);
    if (!el) return;
    var near = thread.getBoundingClientRect().bottom - window.innerHeight < 160;
    var old = el.id ? document.getElementById(el.id) : null;
    if (old && thread.contains(old)) {
      old.replaceWith(el);
    } else {
      thread.appendChild(el);
    }
    if (near) el.scrollIntoView({ block: "nearest", behavior: reduceMotion ? "auto" : "smooth" });
  }

  function setBusy(on) {
    Array.prototype.forEach.call(document.querySelectorAll("[data-start], [data-start-all], [data-start-real]"), function (start) {
      start.disabled = on;
    });
  }

  function end(root, status, html, advanceBatch) {
    if (source) source.close();
    source = null;
    status.innerHTML = html;
    var stop = root.querySelector("[data-stop-run]");
    if (stop) stop.hidden = true;
    setBusy(false);
    if (!advanceBatch) {
      if (batch) endBatch(batch.real ? "Real dataset stopped: " + tally() : undefined);
      return;
    }
    if (batch && batch.real) countReal(html);
    startNextBatch();
  }

  function gone(root, status) {
    end(root, status, "<p>The server no longer has this run, probably because it was restarted. Start the call again. If it was saved, it is on the Dataset page.</p>");
  }

  // The stream closed for good: ask the server whether the run still exists.
  function lost(root, status) {
    fetch(root.dataset.state)
      .then(function (r) {
        if (r.status === 404) return gone(root, status);
        end(root, status, "<p>The connection to the server was lost. The call keeps running there and is saved. Reload this page to watch it.</p>");
      })
      .catch(function () {
        end(root, status, "<p>Cannot reach the server. A call that was running there is saved as far as it got. Reload this page when the server is back.</p>");
      });
  }

  function connect(root) {
    if (source) source.close();
    var thread = root.querySelector("#studio-thread");
    var status = root.querySelector("#studio-status");
    var stop = root.querySelector("[data-stop-run]");
    setBusy(true);
    source = new EventSource(root.dataset.events);
    source.addEventListener("open", function () {
      status.textContent = "The call is being written.";
    });
    ["thinking", "footing", "plan", "turn", "ended", "verdict"].forEach(function (name) {
      source.addEventListener(name, function (e) {
        put(thread, e.data);
      });
    });
    source.addEventListener("failed", function (e) {
      var box = toElement(e.data);
      if (box) status.before(box);
    });
    source.addEventListener("done", function (e) {
      end(root, status, e.data, true);
    });
    source.onerror = function () {
      if (source && source.readyState === EventSource.CLOSED) {
        lost(root, status);
      } else {
        status.textContent = "Reconnecting…";
      }
    };
    if (stop) {
      stop.addEventListener("click", function () {
        stop.disabled = true;
        status.textContent = "Stopping after the step in progress…";
        fetch(root.dataset.stop, { method: "POST" })
          .then(function (r) {
            if (r.status === 404) return gone(root, status);
            if (!r.ok) throw new Error("HTTP " + r.status);
          })
          .catch(function () {
            stop.disabled = false;
            status.textContent = "Could not stop the call. Check the connection and try again.";
          });
      });
    }
    var heading = root.querySelector("#run-h");
    if (heading) heading.focus({ preventScroll: true });
  }

  function connectNew() {
    var root = document.querySelector("[data-studio-run]");
    if (root && !root.dataset.connected) {
      root.dataset.connected = "1";
      connect(root);
    }
  }

  function batchStatus(text) {
    var line = document.getElementById("batch-status");
    if (line) line.textContent = text;
  }

  // The batch is over: the next Start is a single call again, not part of the real dataset.
  function endBatch(message) {
    var ended = batch;
    batch = null;
    var field = document.getElementById("batch");
    if (field) field.value = "";
    if (message !== undefined) batchStatus(message);
    if (ended && ended.real && ended.ids.length) showTokens(ended.ids);
  }

  // The token usage of the calls a real-dataset batch started, above the run.
  function showTokens(ids) {
    var box = document.getElementById("batch-tokens");
    if (!box) return;
    var query = ids.map(function (id) { return "id=" + encodeURIComponent(id); }).join("&");
    box.innerHTML = "<p class='text-[15px] text-muted'>Adding up the batch’s token usage…</p>";
    fetch("/studio/tokens?" + query)
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.text();
      })
      .then(function (html) { box.innerHTML = html; })
      .catch(function () {
        box.innerHTML = "<p class='text-[15px] text-muted'>Could not load the batch’s token usage. Each call’s usage is on the Dataset page.</p>";
      });
  }

  function tally() {
    var counted = batch.approved + batch.rejected;
    var rate = counted ? " (" + Math.round((100 * batch.approved) / counted) + "% approved)" : "";
    return batch.approved + " approved, " + batch.rejected + " rejected" + rate + ".";
  }

  // A call of the real dataset has ended: count it by the tag the server gave it. Stop the batch
  // when the approval rate drops below half, or when the call did not finish (Stop, an error).
  function countReal(html) {
    var t = document.createElement("template");
    t.innerHTML = html;
    var mark = t.content.querySelector("[data-real-result]");
    var result = mark ? mark.dataset.realResult : "";
    var total = batch.size;
    if (mark && mark.dataset.recordId) batch.ids.push(mark.dataset.recordId); // counted or not, it spent tokens
    if (result === "real_approved") {
      batch.approved++;
    } else if (result === "real_rejected") {
      batch.rejected++;
    } else {
      return endBatch("Real dataset stopped after call " + batch.next + " of " + total + ", which did not finish. " + tally());
    }
    if (batch.approved / (batch.approved + batch.rejected) < 0.5) {
      return endBatch("Real dataset stopped after call " + batch.next + " of " + total + ": fewer than half passed the checks. " + tally());
    }
    if (batch.next >= total) endBatch("Real dataset done: all " + total + " calls. " + tally());
  }

  // Run each kind (or each call of the real dataset) in turn, through the same form as a single
  // call. The server still permits only one active run; advancing after its "done" event
  // preserves that.
  function startNextBatch() {
    if (!batch) return endBatch();
    var steps = batch.real ? batch.scenarios : batch.types;
    var total = batch.real ? batch.size : steps.length;
    if (batch.next >= total) return endBatch();
    var form = document.getElementById("studio-form");
    var type = document.getElementById("call_type");
    var scenario = document.getElementById("scenario");
    var field = document.getElementById("batch");
    if (!form || !type || !scenario || !field) return endBatch();
    if (batch.real) {
      type.value = batch.type;
      filterScenarios();
      scenario.value = steps[batch.next % steps.length]; // past the last scenario, start again
      randomCall();
      field.value = "real";
      batchStatus("Real dataset: call " + (batch.next + 1) + " of " + total + ", " + scenario.selectedOptions[0].textContent + ". " + tally());
    } else {
      type.value = steps[batch.next];
      filterScenarios();
      field.value = "";
    }
    batch.next++;
    setBusy(true);
    form.requestSubmit();
  }

  // A random person, a random outcome, the next free seed, no speech.
  function randomCall() {
    document.getElementById("receiver").value = "";
    document.getElementById("outcome").value = "random";
    document.getElementById("seed").value = "";
    document.getElementById("speak").checked = false;
    toggleCustom();
  }

  // Keep the chosen kind and make REAL_BATCH_SIZE calls, going through its scenarios in the
  // list's order and starting again from the first when there are fewer scenarios than calls.
  function startRealDataset() {
    var type = document.getElementById("call_type");
    var scenario = document.getElementById("scenario");
    var ids = ["receiver", "outcome", "seed", "speak", "batch"];
    if (!type || !scenario || ids.some(function (id) { return !document.getElementById(id); })) return;
    var scenarios = Array.prototype.filter
      .call(scenario.options, function (o) { return o.dataset.type === type.value; })
      .map(function (o) { return o.value; });
    if (!scenarios.length) return;
    batch = { real: true, type: type.value, scenarios: scenarios, size: REAL_BATCH_SIZE, next: 0, approved: 0, rejected: 0, ids: [] };
    var tokens = document.getElementById("batch-tokens");
    if (tokens) tokens.innerHTML = "";
    startNextBatch();
  }

  function startAllKinds() {
    var type = document.getElementById("call_type");
    var scenario = document.getElementById("scenario");
    var receiver = document.getElementById("receiver");
    var seed = document.getElementById("seed");
    var speak = document.getElementById("speak");
    if (!type || !scenario || !receiver || !seed || !speak) return;
    var types = Array.prototype.map.call(type.options, function (option) { return option.value; });
    var first = types.indexOf(type.value);
    if (first < 0) return;
    batch = { types: types.slice(first).concat(types.slice(0, first)), next: 0 };
    batchStatus("");
    scenario.value = "";
    receiver.value = "";
    seed.value = "";
    speak.checked = false;
    toggleCustom();
    startNextBatch();
  }

  // The scenario list only shows scenarios of the chosen kind of call.
  function filterScenarios() {
    var type = document.getElementById("call_type");
    var scenario = document.getElementById("scenario");
    if (!type || !scenario) return;
    Array.prototype.forEach.call(scenario.options, function (o) {
      var hide = o.dataset.type && o.dataset.type !== type.value;
      o.hidden = hide;
      o.disabled = hide;
    });
    if (scenario.selectedOptions[0] && scenario.selectedOptions[0].disabled) scenario.value = "";
  }

  // The fields for a described person are shown only when that is chosen.
  function toggleCustom() {
    var who = document.getElementById("receiver");
    var box = document.querySelector("[data-custom]");
    if (!who || !box) return;
    var on = who.value === "__custom__";
    box.hidden = !on;
    Array.prototype.forEach.call(box.querySelectorAll("input, select"), function (f) {
      f.disabled = !on;
    });
  }

  document.addEventListener("change", function (e) {
    if (e.target.id === "call_type") {
      if (batch && batch.real) e.target.value = batch.type; // the kind stays fixed until it ends
      filterScenarios();
    }
    if (e.target.id === "receiver") toggleCustom();
  });
  document.addEventListener("click", function (e) {
    if (e.target.closest("[data-start-all]")) startAllKinds();
    if (e.target.closest("[data-start-real]")) startRealDataset();
  });
  // The server refused the next call of a batch (another run is busy, a form problem): its
  // message is in #form-error, and the batch ends so the buttons work again.
  document.addEventListener("htmx:afterRequest", function (e) {
    if (!batch || !e.target || e.target.id !== "studio-form" || e.detail.successful) return;
    endBatch(batch.real ? "Real dataset stopped: the next call could not start. " + tally() : undefined);
    setBusy(false);
  });
  document.addEventListener("htmx:afterSettle", function (e) {
    if (e.target && e.target.id === "studio-run") connectNew();
  });
  document.addEventListener("DOMContentLoaded", function () {
    filterScenarios();
    toggleCustom();
    connectNew();
  });
})();
