/* Live call page.
 *
 * Browser microphone -> AudioWorklet (16 kHz PCM16, about 100 ms frames) -> WebSocket /ws/live.
 * "Play test call with TTS" feeds the worklet the script's caller lines, spoken by the server's
 * TTS (GET /live/script/{call}/{line}.wav), instead of the microphone, one line per turn.
 * The server answers with JSON events (ready, partial, turn, node, verdict, reply, error, ended)
 * and this file draws them into the same markup the server pages use (the <template> tags in
 * live.html). Decisions are the server's; this file only shows them.
 */
(function () {
  "use strict";

  var root = document.getElementById("live");
  if (!root) return;

  var SENSITIVE = { PHONE: 1, BANK_ACCOUNT: 1, OTP: 1, MONEY: 1, URL: 1 };
  var HIGH_CONF = parseFloat(root.dataset.highConf) || 0.7;
  var TYPES = JSON.parse(root.dataset.types || "{}");
  var IS_DEMO = root.dataset.demo === "true";
  var METER_BARS = 44;
  var WAVE_BARS = 28;
  var MIN_COLUMNS = 7;
  var WIDE = window.matchMedia("(min-width: 1024px)");
  var SCROLL = window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth";

  var $ = function (sel, scope) {
    return (scope || root).querySelector(sel);
  };
  var tpl = function (id) {
    return document.getElementById(id).content.firstElementChild.cloneNode(true);
  };

  var el = {
    slip: $("#live-slip"),
    status: $("[data-status]"),
    timer: $("[data-timer]"),
    meter: $("[data-meter]"),
    error: $("#live-error"),
    start: $('[data-act="start"]'),
    end: $('[data-act="end"]'),
    pause: $('[data-act="pause"]'),
    script: $('[data-act="script"]'), // real mode only
    tts: $('[data-act="tts"]'), // real mode with a test call only
    models: $("[data-models]"), // real mode with a fine-tuned classifier only
    afterEsc: $("[data-after-escalation]"),
    memory: $("[data-memory]"),
    stages: $("[data-stages]"),
    bandSlot: $("[data-band-slot]"),
    replyCard: $("[data-reply-card]"),
    play: $("[data-play]"),
    replyTitle: $("[data-reply-title]"),
    replyText: $("[data-reply-text]"),
    replyMeta: $("[data-reply-meta]"),
    replyTime: $("[data-reply-time]"),
    wave: $("[data-wave]"),
    audio: $("[data-audio]"),
    strip: $("[data-strip]"),
    rows: $("[data-rows]"),
    rowsEmpty: $("[data-rows-empty]"),
  };

  // ---- state ----------------------------------------------------------------------------
  var S;
  function freshState() {
    return {
      ws: null,
      stream: null,
      ctx: null,
      node: null,
      source: null,
      sink: null,
      listening: false,
      userPaused: false,
      replyPlaying: false,
      serverPaused: false,
      ending: false,
      startedAt: 0,
      timerId: 0,
      turns: [], // { n, row, col, text, speaker, start, level, replied }
      byServerTurn: {},
      partialRow: null,
      entities: [],
      escN: null,
      escLevel: null,
      escType: null,
      escConf: 0,
      summary: "",
      bandKind: null,
      meterLevels: [],
      replyUrl: null,
      tts: false, // the TTS reads the script instead of the microphone
      ttsLine: 0, // the script line the TTS is on
      ttsGap: false, // between two TTS lines: the server is paused so the caller's turn closes
      doneUpTo: 0, // the last turn the server finished checking
    };
  }
  S = freshState();

  // ---- small helpers --------------------------------------------------------------------
  function fmt(seconds) {
    var s = Math.max(0, Math.floor(seconds));
    return Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0");
  }
  function showError(message) {
    el.error.textContent = "";
    if (!message) return;
    var p = document.createElement("p");
    p.className = "rounded-[12px] border border-alarm-line bg-alarm-tint px-4 py-3 text-[16px] leading-[1.6] text-ink";
    p.setAttribute("role", "alert");
    p.textContent = message;
    el.error.appendChild(p);
  }
  function slot(node, name) {
    return node.querySelector('[data-slot="' + name + '"]');
  }
  // Restart a CSS animation that is triggered by a class (see "motion" in app.css).
  function replay(node, cls) {
    node.classList.remove(cls);
    void node.offsetWidth;
    node.classList.add(cls);
  }

  // ---- level meter ----------------------------------------------------------------------
  function buildMeter() {
    el.meter.textContent = "";
    for (var i = 0; i < METER_BARS; i++) {
      var b = document.createElement("i");
      b.setAttribute("data-off", "");
      el.meter.appendChild(b);
    }
  }
  function drawMeter(level) {
    // RMS is small for speech; a square-root curve spreads it over the bar height.
    var h = Math.min(1, Math.sqrt(level) * 1.6);
    S.meterLevels.push(h);
    if (S.meterLevels.length > METER_BARS) S.meterLevels.shift();
    var bars = el.meter.children;
    var offset = METER_BARS - S.meterLevels.length;
    for (var i = 0; i < METER_BARS; i++) {
      var v = i < offset ? 0 : S.meterLevels[i - offset];
      bars[i].style.transform = "scaleY(" + Math.max(6, Math.round(v * 56)) / 56 + ")";
      if (v > 0.02) bars[i].removeAttribute("data-off");
      else bars[i].setAttribute("data-off", "");
    }
  }
  function flatMeter() {
    S.meterLevels = [];
    var bars = el.meter.children;
    for (var i = 0; i < bars.length; i++) {
      bars[i].style.transform = "";
      bars[i].setAttribute("data-off", "");
    }
  }

  // ---- stages ---------------------------------------------------------------------------
  function setStage(n, state, desc, right, tone) {
    var li = el.stages.querySelector('[data-stage="' + n + '"]');
    if (!li) return;
    var before = li.getAttribute("data-state");
    li.setAttribute("data-state", state);
    if (state === "done" && before !== "done") replay(li, "is-new"); // the check draws once
    if (desc != null) slot(li, "desc").textContent = desc;
    var r = slot(li, "right");
    r.textContent = right || "";
    if (tone) r.setAttribute("data-tone", tone);
    else r.removeAttribute("data-tone");
  }
  function resetStages() {
    setStage(1, "idle", "Streaming, partial text every 0.5 s", "");
    setStage(2, "idle", "Turn closes on silence", "");
    setStage(3, "idle", "Checks each turn with the turns before it", "");
    setStage(4, "idle", "Fixed rules. Reply or warn.", "");
  }
  function stagesForListening() {
    setStage(1, S.listening ? "busy" : "idle", null, S.listening ? "Live" : "Stopped", S.listening ? "live" : "");
  }

  // ---- strip ----------------------------------------------------------------------------
  function buildStrip() {
    el.strip.textContent = "";
    for (var n = 1; n <= MIN_COLUMNS; n++) el.strip.appendChild(makeCol(n));
  }
  function makeCol(n) {
    var col = tpl("tpl-col");
    col.setAttribute("data-turn", String(n));
    slot(col, "tlabel").textContent = "T" + n;
    return col;
  }
  function colFor(n) {
    var col = el.strip.querySelector('[data-turn="' + n + '"]');
    while (!col) {
      col = makeCol(el.strip.children.length + 1);
      el.strip.appendChild(col);
      col = el.strip.querySelector('[data-turn="' + n + '"]');
    }
    return col;
  }
  function setBar(col, state, height) {
    var well = col.querySelector(".well");
    // partial text arrives every ~0.5 s: keep the same "listening" bar so its breathing does not restart
    if (state === "live" && well.querySelector('.bar[data-state="live"]')) return;
    var bar = tpl("tpl-bar-" + state);
    if (height) bar.style.setProperty("--h", height + "%");
    if (state === "normal" || state === "scam" || state === "held") bar.classList.add("is-new"); // grows from the baseline
    well.replaceChildren(bar);
  }
  function barHeight(text) {
    return 28 + Math.floor((Math.min(text.length, 100) * 3) / 10);
  }

  // ---- band -----------------------------------------------------------------------------
  function setSlots(band, values) {
    Object.keys(values).forEach(function (k) {
      var n = slot(band, k);
      if (n) n.textContent = values[k];
    });
  }
  function showBand(kind, values, fresh) {
    var band = el.bandSlot.querySelector("[data-band]");
    if (!band || S.bandKind !== kind) {
      band = tpl("tpl-band-" + kind);
      // The slot is the live region; a band inserted with its own would be read twice.
      band.removeAttribute("role");
      band.removeAttribute("aria-live");
      band.classList.add("is-arriving"); // wash + rise, only when the kind changes
      if (fresh) band.classList.add("is-fresh");
      el.bandSlot.replaceChildren(band);
      S.bandKind = kind;
    }
    setSlots(band, values);
  }
  function renderBand() {
    var checked = S.turns.filter(function (t) {
      return t.level != null;
    }).length;
    if (S.escN != null) {
      var scam = S.escLevel === "scam";
      var conf = S.escConf >= HIGH_CONF ? "high" : S.escConf > 0 ? "low" : "not known";
      showBand(
        scam ? "scam" : "unsure",
        {
          word: scam ? "SCAM" : "UNSURE",
          subtype: scam ? TYPES[S.escType] || TYPES.null || "Type not known" : "Might be a scam. Type not known",
          instruction: "Hang up on your phone too. Do not transfer money.",
          inline: "Escalated at turn " + S.escN + ". Callas stopped replying.",
          "fact-label": "Escalated at",
          "fact-big": "Turn " + S.escN,
          "fact-note": "of " + S.turns.length + " so far. Stays escalated until the call ends.",
          "fact-extra": "Confidence: " + conf + ". No reply was spoken.",
        },
        true
      );
      return;
    }
    if (checked === 0) return;
    var next = checked + 1;
    var note = S.listening
      ? "Turn " + next + " is being heard now. The verdict can still change."
      : "The call is over. No scam signs found.";
    showBand("safe", {
      word: "NORMAL",
      subtype: "No scam signs in " + checked + (checked === 1 ? " turn" : " turns"),
      instruction: "Callas may answer. Every reply passes the leak check first.",
      inline: note,
      "fact-label": "Checked",
      "fact-big": checked + (checked === 1 ? " turn" : " turns"),
      "fact-note": note,
      "fact-extra": "",
    });
  }

  // ---- transcript rows ------------------------------------------------------------------
  function fillText(node, text, entities) {
    var spans = [];
    entities.forEach(function (e) {
      if (!e.text) return;
      var i = text.indexOf(e.text);
      if (i >= 0) spans.push({ s: i, e: i + e.text.length, label: e.label });
    });
    spans.sort(function (a, b) {
      return a.s - b.s;
    });
    var frag = document.createDocumentFragment();
    var pos = 0;
    spans.forEach(function (sp) {
      if (sp.s < pos) return;
      if (sp.s > pos) frag.appendChild(document.createTextNode(text.slice(pos, sp.s)));
      var span = document.createElement("span");
      span.className = "ent";
      span.setAttribute("data-tone", SENSITIVE[sp.label] ? "alarm" : "ink");
      span.setAttribute("data-label", sp.label);
      span.textContent = text.slice(sp.s, sp.e);
      frag.appendChild(span);
      pos = sp.e;
    });
    if (pos < text.length) frag.appendChild(document.createTextNode(text.slice(pos)));
    node.replaceChildren(frag);
  }
  function setTag(row, text, tone) {
    var tag = slot(row, "tag");
    tag.textContent = text || "";
    if (tone) tag.setAttribute("data-tone", tone);
    else tag.removeAttribute("data-tone");
  }
  function keepInView(row) {
    if (WIDE.matches && S.listening) row.scrollIntoView({ block: "nearest", behavior: SCROLL });
  }
  function ensurePartialRow() {
    if (S.partialRow) return S.partialRow;
    var row = tpl("tpl-row");
    var n = S.turns.length + 1;
    row.setAttribute("data-turn", String(n));
    slot(row, "n").textContent = String(n);
    slot(row, "speaker").textContent = "";
    row.setAttribute("data-speaker", "");
    slot(row, "time").textContent = fmt(S.listening ? (performance.now() - S.startedAt) / 1000 : 0);
    setTag(row, "Hearing", "");
    row.classList.add("is-new");
    el.rows.appendChild(row);
    el.rowsEmpty.hidden = true;
    S.partialRow = row;
    keepInView(row);
    return row;
  }
  function drawPartial(text) {
    var row = ensurePartialRow();
    var p = slot(row, "text");
    p.textContent = text;
    var caret = document.createElement("span");
    caret.className = "caret";
    caret.setAttribute("aria-hidden", "true");
    p.appendChild(caret);
    var n = S.turns.length + 1;
    setBar(colFor(n), "live");
    setStage(1, "busy", null, "Live", "live");
  }
  function whoLabel(speaker) {
    return speaker === "A" ? "You" : "Caller";
  }
  function onTurn(ev) {
    var n = S.turns.length + 1;
    var row = S.partialRow || (ensurePartialRow(), S.partialRow);
    S.partialRow = null;
    row.setAttribute("data-state", "pending");
    row.setAttribute("data-speaker", ev.speaker);
    slot(row, "speaker").textContent = ev.speaker.charAt(0);
    slot(row, "time").textContent = fmt(ev.start);
    setTag(row, "Reading", "");
    var t = { n: n, row: row, col: colFor(n), text: ev.text, speaker: ev.speaker, level: null };
    S.turns.push(t);
    S.byServerTurn[ev.turn] = t;
    fillText(slot(row, "text"), ev.text, S.entities);
    slot(t.col, "who").textContent = ev.speaker === "Caller" ? "" : whoLabel(ev.speaker);
    setBar(t.col, "idle");
    setStage(2, "done", null, "Done");
    setStage(3, "busy", "Turn " + n + (n > 1 ? " · checking with turns 1–" + n : " · checking"), "…");
    markScript();
  }
  // The read-aloud script: lines already read fade, the next one is marked. Every turn the
  // microphone hears is the caller, so the caller's turn count is the number of lines read.
  function markScript() {
    var read = S.tts
      ? S.ttsLine
      : S.turns.filter(function (t) {
          return t.speaker !== "A";
        }).length;
    var lines = document.querySelectorAll("[data-script-lines] li");
    lines.forEach(function (li, i) {
      li.toggleAttribute("data-read", i < read);
      if (i === read) li.setAttribute("aria-current", "step");
      else li.removeAttribute("aria-current");
    });
    if (read > 0 && lines[read]) lines[read].scrollIntoView({ block: "nearest", behavior: SCROLL });
  }
  function turnFor(ev) {
    return S.byServerTurn[ev.turn] || S.turns[S.turns.length - 1];
  }
  function rehighlight() {
    S.turns.forEach(function (t) {
      fillText(slot(t.row, "text"), t.text, S.entities);
    });
  }
  function onVerdict(ev) {
    var t = turnFor(ev);
    if (!t) return;
    (ev.entities || []).forEach(function (e) {
      if (
        !S.entities.some(function (k) {
          return k.text === e.text && k.label === e.label;
        })
      )
        S.entities.push(e);
    });
    rehighlight();
    S.summary = ev.summary || S.summary;
    t.level = ev.label_scam ? (ev.confidence >= HIGH_CONF ? "scam" : "unsure") : "safe";
    var first = ev.escalated && S.escN == null;
    if (first) {
      S.escN = t.n;
      S.escLevel = t.level === "safe" ? "unsure" : t.level;
      S.escType = ev.label_type;
      S.escConf = ev.confidence || 0;
    }
    if (ev.escalated && !first) {
      t.row.setAttribute("data-state", "held");
      setTag(t.row, "Held", "");
      setBar(t.col, "held");
    } else if (first) {
      t.row.setAttribute("data-state", "scam");
      setTag(t.row, "Escalated here", "alarm");
      setBar(t.col, "scam");
      var rowPen = tpl("tpl-pen-row");
      t.row.classList.add("is-fresh");
      slot(t.row, "text").appendChild(rowPen); // circle the sentence, not the badge
      var colPen = tpl("tpl-pen-col");
      t.col.classList.add("is-fresh");
      t.col.querySelector(".well").appendChild(colPen); // circle the bar, not its labels
      slot(t.col, "tlabel").classList.add("text-alarm");
      el.afterEsc.hidden = false;
    } else {
      t.row.setAttribute("data-state", "normal");
      setTag(t.row, "", "");
      setBar(t.col, "normal", barHeight(t.text));
    }
    var n = t.n;
    setStage(3, "done", "Checked turns 1–" + n, "Done");
    if (S.escN != null) setStage(4, "scam", "Scam found at turn " + S.escN + ". Escalated. No reply.", "Done");
    else setStage(4, "safe", "Normal so far. Reply allowed.", "Done");
    renderBand();
  }
  function onNode(ev) {
    var t = turnFor(ev);
    if (!t) return;
    if (ev.node === "analyze") setStage(3, "busy", null, "…");
    if (ev.node === "remember" && ev.data && ev.data.turn && ev.data.turn.text !== t.text) {
      t.text = ev.data.turn.text; // the cleaned text (normalized, maybe autocorrected)
      fillText(slot(t.row, "text"), t.text, S.entities);
    }
  }

  // ---- reply audio ----------------------------------------------------------------------
  function buildWave() {
    el.wave.textContent = "";
    for (var i = 0; i < WAVE_BARS; i++) {
      var b = document.createElement("i");
      b.style.height = 6 + Math.round(Math.abs(Math.sin(i * 1.7) * Math.cos(i * 0.6)) * 30) + "px";
      b.setAttribute("data-off", "");
      el.wave.appendChild(b);
    }
  }
  function drawWave(fraction) {
    var on = Math.round(fraction * WAVE_BARS);
    for (var i = 0; i < el.wave.children.length; i++) {
      if (i < on) el.wave.children[i].removeAttribute("data-off");
      else el.wave.children[i].setAttribute("data-off", "");
    }
  }
  function setPlayIcon(playing) {
    el.play.setAttribute("aria-label", playing ? "Pause the reply" : "Play the reply");
    el.play.dataset.playing = playing ? "true" : "false";
  }
  function setProgress(fraction) {
    el.play.style.setProperty("--p", String(Math.round(Math.min(1, Math.max(0, fraction)) * 1000) / 10));
  }
  function bytesFromBase64(b64) {
    var bin = atob(b64);
    var bytes = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return bytes;
  }
  function addReplyRow(t, text) {
    var row = tpl("tpl-row");
    row.removeAttribute("data-turn");
    row.setAttribute("data-speaker", "Callas");
    row.setAttribute("data-state", "reply");
    var badge = slot(row, "speaker");
    badge.textContent = "";
    badge.appendChild(tpl("tpl-callas-mark"));
    $(".trow-turn", row).textContent = "Callas";
    slot(row, "time").textContent = "Reply to turn " + t.n;
    slot(row, "text").textContent = text;
    setTag(row, "", "");
    t.row.after(row);
    keepInView(row);
  }
  function onReply(ev) {
    var t = turnFor(ev);
    if (t) {
      t.replied = true;
      setTag(t.row, "Reply sent", "safe");
      addReplyRow(t, ev.text);
    }
    el.replyCard.hidden = false;
    el.replyTitle.textContent = "Callas answered after turn " + (t ? t.n : ev.turn);
    el.replyText.textContent = "“" + ev.text + "”";
    el.replyMeta.textContent = IS_DEMO
      ? "Scripted demo reply · silent audio"
      : (ev.template_used ? "Fixed reply passed the leak check" : "LLM reply passed the leak check") +
        " · FastThaiG2P + PyThaiTTS";
    if (S.replyUrl) URL.revokeObjectURL(S.replyUrl);
    setProgress(0);
    if (!ev.audio_wav_b64) {
      S.replyUrl = null;
      el.audio.removeAttribute("src");
      return;
    }
    S.replyUrl = URL.createObjectURL(new Blob([bytesFromBase64(ev.audio_wav_b64)], { type: "audio/wav" }));
    el.audio.src = S.replyUrl;
    // Callas' own voice must not be transcribed: pause the server while it plays.
    S.replyPlaying = true;
    syncPause();
    var started = el.audio.play();
    if (started && started.catch) {
      started.catch(function () {
        S.replyPlaying = false; // the browser refused to play it; the play button still works
        syncPause();
      });
    }
  }
  el.audio.addEventListener("play", function () {
    setPlayIcon(true);
  });
  el.audio.addEventListener("pause", function () {
    setPlayIcon(false);
  });
  el.audio.addEventListener("ended", function () {
    setPlayIcon(false);
    setProgress(1);
    drawWave(1);
    S.replyPlaying = false;
    syncPause();
  });
  el.audio.addEventListener("timeupdate", function () {
    var d = el.audio.duration || 0;
    el.replyTime.textContent = fmt(el.audio.currentTime) + " / " + fmt(d);
    drawWave(d ? el.audio.currentTime / d : 0);
    setProgress(d ? el.audio.currentTime / d : 0);
  });
  el.audio.addEventListener("loadedmetadata", function () {
    el.replyTime.textContent = "0:00 / " + fmt(el.audio.duration || 0);
  });
  el.play.addEventListener("click", function () {
    if (!el.audio.src) return;
    if (el.audio.paused) {
      if (el.audio.ended) el.audio.currentTime = 0;
      el.audio.play();
    } else {
      el.audio.pause();
    }
  });

  // ---- pause / resume (mic button and reply playback) ------------------------------------
  function syncPause() {
    var want = S.userPaused || S.replyPlaying || S.ttsGap;
    if (want === S.serverPaused || !S.ws || S.ws.readyState !== WebSocket.OPEN) return;
    S.serverPaused = want;
    S.ws.send(JSON.stringify({ type: want ? "pause" : "resume" }));
    if (want) flatMeter();
    updateStatus();
  }
  function updateStatus() {
    var label = !S.listening
      ? "Not listening"
      : S.userPaused
        ? "Mic paused"
        : S.replyPlaying
          ? "Callas is speaking"
          : S.ttsGap
            ? "Callas is checking"
            : S.tts
              ? "TTS caller speaking"
              : "Listening";
    el.status.textContent = label;
    var hearing = S.listening && !S.userPaused && !S.replyPlaying && !S.ttsGap;
    el.slip.setAttribute("data-listening", hearing ? "true" : "false");
  }

  // ---- WebSocket events -----------------------------------------------------------------
  function onEvent(ev) {
    switch (ev.type) {
      case "ready":
        break;
      case "partial":
        drawPartial(ev.text);
        break;
      case "turn":
        onTurn(ev);
        break;
      case "node":
        onNode(ev);
        break;
      case "verdict":
        onVerdict(ev);
        break;
      case "reply":
        onReply(ev);
        break;
      case "error":
        showError(ev.message || "Something went wrong during the call.");
        break;
      case "done":
        S.doneUpTo = Math.max(S.doneUpTo, ev.turn);
        break;
      case "ended":
        finish(true);
        break;
    }
  }

  // ---- start / end ----------------------------------------------------------------------
  function micMessage(err) {
    var name = err && err.name;
    if (name === "NotAllowedError" || name === "SecurityError")
      return "Microphone blocked. Allow it in the browser bar, then try again.";
    if (name === "NotFoundError" || name === "OverconstrainedError")
      return "No microphone found. Plug one in, then try again.";
    if (name === "NotReadableError") return "The microphone is in use by another app. Close it, then try again.";
    return "The microphone could not be opened. Check the browser bar, then try again.";
  }
  function setControls(running) {
    el.start.hidden = running;
    el.end.hidden = !running;
    el.pause.hidden = !running || S.tts;
    if (el.tts) el.tts.hidden = running;
    el.pause.textContent = "Pause mic";
    // A new script mid-call would no longer match the lines already spoken.
    if (el.script) el.script.disabled = running;
    // The models are loaded when the call starts; switching mid-call would not apply.
    if (el.models) el.models.disabled = running;
  }

  async function start(useTts) {
    showError("");
    var needMic = !useTts;
    if ((needMic && (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia)) || !window.AudioWorkletNode || !window.WebSocket) {
      showError("This browser cannot record audio here. Use a current Chrome, Edge, Firefox or Safari over https or localhost.");
      return;
    }
    setStartDisabled(true);
    resetCall();
    S.tts = useTts;
    var stream = null;
    if (needMic) {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
        });
      } catch (err) {
        setStartDisabled(false);
        showError(micMessage(err));
        return;
      }
    }
    S.stream = stream;
    try {
      var Ctx = window.AudioContext || window.webkitAudioContext;
      S.ctx = new Ctx();
      await S.ctx.audioWorklet.addModule(root.dataset.worklet || "/static/js/pcm-worklet.js");
      if (stream) {
        S.source = S.ctx.createMediaStreamSource(stream);
      } else {
        // Silence between the lines keeps frames flowing, so each line's last frame is sent.
        S.source = S.ctx.createConstantSource();
        S.source.offset.value = 0;
        S.source.start();
      }
      S.node = new AudioWorkletNode(S.ctx, "pcm16-downsampler", {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        outputChannelCount: [1],
        processorOptions: { targetRate: 16000, frameMs: 100 },
      });
      S.sink = S.ctx.createGain();
      S.sink.gain.value = 0;
      S.source.connect(S.node);
      S.node.connect(S.sink);
      S.sink.connect(S.ctx.destination);
    } catch (err) {
      cleanupAudio();
      setStartDisabled(false);
      showError("The audio engine could not start in this browser. Try Chrome or Edge.");
      return;
    }
    S.node.port.onmessage = function (m) {
      var d = m.data;
      if (d.level != null) {
        if (!S.serverPaused) drawMeter(d.level);
      } else if (d.pcm && S.ws && S.ws.readyState === WebSocket.OPEN && !S.serverPaused) {
        S.ws.send(d.pcm);
      }
    };
    var url = (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws/live";
    if (el.models) {
      url += "?" + new URLSearchParams({ gate: $("#model-gate").value, type: $("#model-type").value });
    }
    var ws = new WebSocket(url);
    ws.binaryType = "arraybuffer";
    S.ws = ws;
    ws.onopen = function () {
      S.listening = true;
      S.startedAt = performance.now();
      S.timerId = window.setInterval(function () {
        el.timer.textContent = fmt((performance.now() - S.startedAt) / 1000);
      }, 250);
      setControls(true);
      setStartDisabled(false);
      stagesForListening();
      updateStatus();
      if (S.tts) playTestCall();
    };
    ws.onmessage = function (m) {
      try {
        onEvent(JSON.parse(m.data));
      } catch (e) {
        /* ignore a malformed event */
      }
    };
    ws.onerror = function () {
      if (!S.listening) {
        showError("Cannot reach the live call server. Check that it is running, then try again.");
        finish(false);
      }
    };
    ws.onclose = function () {
      if (S.listening && !S.ending) {
        showError("The connection to the server closed. Start the call again.");
        finish(false);
      }
    };
  }

  function setStartDisabled(disabled) {
    el.start.disabled = disabled;
    if (el.tts) el.tts.disabled = disabled;
  }

  // ---- TTS caller -------------------------------------------------------------------------
  // Each line: play it, pause the server (which closes the caller's turn), wait until the
  // server has checked that turn and Callas has finished speaking, then go on.
  async function playTestCall() {
    var call = S;
    var list = document.querySelector("[data-script-lines]");
    var count = list ? list.children.length : 0;
    var alive = function () {
      return S === call && call.listening;
    };
    var next = count ? fetchLine(list.dataset.callId, 0) : null;
    for (var i = 0; i < count; i++) {
      var buffer;
      try {
        buffer = await next;
      } catch (err) {
        if (alive()) showError("The TTS could not read line " + (i + 1) + " of the script. Try another test call.");
        break;
      }
      if (i + 1 < count) next = fetchLine(list.dataset.callId, i + 1);
      await waitFor(alive, function () {
        return !call.replyPlaying;
      });
      if (!alive()) return;
      call.ttsLine = i;
      markScript();
      call.ttsGap = false;
      syncPause();
      updateStatus();
      await playBuffer(buffer);
      await waitFor(alive, null, 400); // the line's last frames reach the server first
      if (!alive()) return;
      var turnsBefore = call.turns.length;
      call.ttsGap = true;
      syncPause();
      // No turn after 5 s: the speech recogniser heard nothing in this line.
      await waitFor(alive, function () {
        return call.turns.length > turnsBefore;
      }, 5000);
      await waitFor(alive, function () {
        return call.doneUpTo >= call.turns.length;
      }, 60000);
    }
    if (!alive()) return;
    call.ttsLine = count;
    markScript();
    await waitFor(alive, function () {
      return !call.replyPlaying;
    });
    if (alive()) endCall();
  }
  function fetchLine(callId, i) {
    var ctx = S.ctx;
    var p = fetch("/live/script/" + encodeURIComponent(callId) + "/" + i + ".wav")
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.arrayBuffer();
      })
      .then(function (bytes) {
        return ctx.decodeAudioData(bytes);
      });
    p.catch(function () {}); // reported when the loop reaches this line
    return p;
  }
  function playBuffer(buffer) {
    return new Promise(function (resolve) {
      var line = S.ctx.createBufferSource();
      line.buffer = buffer;
      line.connect(S.node); // to the server
      line.connect(S.ctx.destination); // and to the speakers, so you hear the caller
      line.onended = function () {
        line.disconnect();
        resolve();
      };
      line.start();
    });
  }
  // Resolves once done() is true (or after `ms`, when given), or the call is over.
  function waitFor(alive, done, ms) {
    var waited = 0;
    return new Promise(function (resolve) {
      var id = window.setInterval(function () {
        waited += 100;
        if (!alive() || (done && done()) || (ms && waited >= ms)) {
          window.clearInterval(id);
          resolve();
        }
      }, 100);
    });
  }

  function cleanupAudio() {
    try {
      if (S.node) S.node.port.onmessage = null;
      if (S.source) S.source.disconnect();
      if (S.source && S.source.stop) S.source.stop();
      if (S.node) S.node.disconnect();
    } catch (e) {
      /* already disconnected */
    }
    if (S.stream) S.stream.getTracks().forEach(function (t) { t.stop(); });
    if (S.ctx && S.ctx.state !== "closed") S.ctx.close().catch(function () {});
    S.stream = S.ctx = S.node = S.source = S.sink = null;
  }

  function finish(clean) {
    if (S.timerId) window.clearInterval(S.timerId);
    S.timerId = 0;
    var wasListening = S.listening;
    S.listening = false;
    S.replyPlaying = false;
    cleanupAudio();
    if (S.ws) {
      S.ws.onclose = null;
      try {
        S.ws.close();
      } catch (e) {
        /* closed already */
      }
      S.ws = null;
    }
    if (S.partialRow) {
      S.partialRow.remove();
      S.partialRow = null;
      var live = el.strip.querySelector('.bar[data-state="live"]');
      if (live) setBar(live.closest("[data-turn]"), "idle");
    }
    setControls(false);
    setStartDisabled(false);
    el.start.textContent = "Start a new call";
    el.memory.textContent = clean ? "The call has ended. Its memory was deleted." : "The call stopped. Its memory was deleted.";
    flatMeter();
    updateStatus();
    stagesForListening();
    if (wasListening) renderBand();
    S.ending = false;
  }

  function endCall() {
    if (!S.ws) return finish(true);
    S.ending = true;
    try {
      S.ws.send(JSON.stringify({ type: "end" }));
    } catch (e) {
      return finish(true);
    }
    // The server answers "ended". If it does not, close anyway.
    window.setTimeout(function () {
      if (S.ending) finish(true);
    }, 2500);
  }

  function resetCall() {
    el.rows.textContent = "";
    el.rowsEmpty.hidden = false;
    buildStrip();
    resetStages();
    el.replyCard.hidden = true;
    el.afterEsc.hidden = true;
    el.timer.textContent = "0:00";
    el.memory.textContent = "Ending the call deletes its memory.";
    if (S.replyUrl) URL.revokeObjectURL(S.replyUrl);
    el.audio.removeAttribute("src");
    S = freshState();
    el.bandSlot.replaceChildren(placeholderBand());
    markScript();
  }
  var placeholderHtml = el.bandSlot.innerHTML;
  function placeholderBand() {
    var d = document.createElement("div");
    d.innerHTML = placeholderHtml;
    return d.firstElementChild;
  }

  el.start.addEventListener("click", function () {
    start(false);
  });
  if (el.tts)
    el.tts.addEventListener("click", function () {
      start(true);
    });
  el.end.addEventListener("click", endCall);
  el.pause.addEventListener("click", function () {
    S.userPaused = !S.userPaused;
    el.pause.textContent = S.userPaused ? "Resume mic" : "Pause mic";
    syncPause();
    updateStatus();
  });
  window.addEventListener("pagehide", function () {
    if (S.ws) {
      try {
        S.ws.send(JSON.stringify({ type: "end" }));
      } catch (e) {
        /* closing anyway */
      }
    }
  });

  document.body.addEventListener("htmx:afterSwap", function (e) {
    if (e.target.id === "live-script") markScript();
  });

  buildMeter();
  buildWave();
  markScript();
  flatMeter();
  updateStatus();
})();
