/* Small page behaviour shared by the pages. No libraries beyond htmx. */
(function () {
  "use strict";

  // htmx does not swap 4xx answers by default. The server sends form problems as 422 (and
  // similar) with a short message and HX-Retarget #form-error, so let those be swapped.
  document.addEventListener("htmx:beforeSwap", function (evt) {
    var status = evt.detail.xhr.status;
    if (status >= 400 && status < 600 && evt.detail.xhr.getResponseHeader("HX-Retarget")) {
      evt.detail.shouldSwap = true;
      evt.detail.isError = false;
    }
  });

  document.addEventListener("htmx:beforeRequest", function () {
    var slot = document.getElementById("form-error");
    if (slot) slot.textContent = "";
  });

  document.addEventListener("htmx:sendError", function () {
    var slot = document.getElementById("form-error");
    if (slot) slot.textContent = "Cannot reach the server. Check that it is running, then try again.";
  });

  // ---- message page ---------------------------------------------------------------------
  var msgForm = document.getElementById("message-form");
  if (msgForm) {
    var text = document.getElementById("text");
    var count = msgForm.querySelector("[data-count]");
    var update = function () {
      count.textContent = String(text.value.length);
    };
    text.addEventListener("input", update);
    msgForm.querySelector("[data-clear]").addEventListener("click", function () {
      text.value = "";
      update();
      var slot = document.getElementById("form-error");
      if (slot) slot.textContent = "";
      text.focus();
    });
    msgForm.querySelector("[data-fill]").addEventListener("click", function (e) {
      text.value = e.currentTarget.getAttribute("data-fill");
      update();
      text.focus();
    });
    update();
  }

  // ---- call page ------------------------------------------------------------------------
  var callForm = document.getElementById("call-form");
  if (callForm) {
    var input = document.getElementById("audio");
    var nameEl = callForm.querySelector("[data-file-name]");
    var metaEl = callForm.querySelector("[data-file-meta]");
    var maxBytes = Number(input.getAttribute("data-max-mb")) * 1024 * 1024;
    var showSize = function (bytes) {
      return bytes >= 1048576 ? (bytes / 1048576).toFixed(1) + " MB" : Math.max(1, Math.round(bytes / 1024)) + " KB";
    };

    callForm.addEventListener("click", function (e) {
      if (e.target.closest("[data-pick]")) input.click();
    });

    // Stop a too-large file before it is sent, with the same message the server uses.
    input.addEventListener("change", function () {
      var f = input.files && input.files[0];
      if (!f) return;
      nameEl.textContent = f.name;
      metaEl.textContent = showSize(f.size);
    });
    callForm.addEventListener(
      "htmx:confirm",
      function (e) {
        var f = input.files && input.files[0];
        if (f && f.size > maxBytes) {
          e.preventDefault();
          document.getElementById("form-error").textContent =
            "The recording is larger than 20 MB. Use a shorter one.";
        }
      },
      true
    );

    document.addEventListener("htmx:afterRequest", function (e) {
      if (e.detail.elt !== callForm || !e.detail.successful) return;
      callForm.querySelector('[data-when="idle"]').hidden = true;
      callForm.querySelector('[data-when="ran"]').hidden = false;
    });
  }
})();
