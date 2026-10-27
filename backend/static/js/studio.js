/* /studio: show a call as the server writes it.
 * The server sends server-sent events whose data is a piece of HTML: "thinking", "plan", "turn"
 * and "verdict" are list items for the thread (an item with the id of an earlier one replaces it,
 * which is how a "thinking" bubble becomes the message), "failed" is an error message and "done"
 * ends the run. A plain EventSource is used instead of the htmx sse extension because an event
 * has to replace one element by id. Closing the tab does not stop the run; the server saves it. */
(function () {
  "use strict";

  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var source = null;
  var batch = null;

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
    Array.prototype.forEach.call(document.querySelectorAll("[data-start], [data-start-all]"), function (start) {
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
    if (advanceBatch) startNextBatch();
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
    ["thinking", "plan", "turn", "ended", "verdict"].forEach(function (name) {
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
        batch = null; // Stop ends a "Generate all six kinds" run too, not just this call
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

  // Run each kind in turn, through the same form as a single call. The server still permits
  // only one active run; advancing after its "done" event preserves that contract.
  function startNextBatch() {
    if (!batch || batch.next >= batch.types.length) {
      batch = null;
      return;
    }
    var form = document.getElementById("studio-form");
    var type = document.getElementById("call_type");
    if (!form || !type) {
      batch = null;
      return;
    }
    type.value = batch.types[batch.next++];
    filterScenarios();
    setBusy(true);
    form.requestSubmit();
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
    if (e.target.id === "call_type") filterScenarios();
    if (e.target.id === "receiver") toggleCustom();
  });
  document.addEventListener("click", function (e) {
    if (e.target.closest("[data-start-all]")) startAllKinds();
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
