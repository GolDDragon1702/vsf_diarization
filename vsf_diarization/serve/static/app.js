"use strict";

const $ = (id) => document.getElementById(id);
const PALETTE = ["#6366f1", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6", "#06b6d4", "#ec4899", "#84cc16"];
const speakerColor = {};
function colorFor(spk) {
  if (!(spk in speakerColor))
    speakerColor[spk] = PALETTE[Object.keys(speakerColor).length % PALETTE.length];
  return speakerColor[spk];
}
const initials = (s) => (s.match(/\d+/) ? s.match(/\d+/)[0] : s.slice(0, 2)).toString();
const fmt = (t) => {
  const m = Math.floor(t / 60), s = (t % 60).toFixed(1).padStart(4, "0");
  return `${m}:${s}`;
};

// ── State + transcript rendering ───────────────────────────────────────
const transcript = $("transcript");
const state = { turns: [], duration: 0, audioName: null, audioURL: null };

function turnNode(t, i) {
  const c = colorFor(t.speaker);
  const pending = t.text === "..." || !t.text;
  const el = document.createElement("div");
  el.className = "turn"; el.id = "turn-" + i;
  el.innerHTML = `
    <div class="avatar" style="background:${c}" title="Bấm để đổi tên">${initials(t.speaker)}</div>
    <div class="bubble ${pending ? "pending" : ""}" style="border-left-color:${c}">
      <div class="meta"><span class="spk" style="color:${c}">${escapeHtml(t.speaker)}</span>
        <span class="time">${fmt(t.start)} → ${fmt(t.end)}</span></div>
      <div class="text">${pending ? "…" : escapeHtml(t.text)}</div>
    </div>`;
  el.querySelector(".avatar").onclick = (e) => { e.stopPropagation(); renameSpeaker(t.speaker); };
  el.querySelector(".bubble").onclick = () => seekTo(t.start);
  return el;
}

function addTurn(t) {
  state.turns.push(t);
  state.duration = Math.max(state.duration, t.end);
  transcript.querySelector(".empty")?.remove();
  const el = turnNode(t, state.turns.length - 1);
  const interim = $("interim");
  if (interim) transcript.insertBefore(el, interim);   // giữ caption "đang nghe" ở cuối
  else transcript.appendChild(el);
  transcript.scrollTop = transcript.scrollHeight;
  renderTimeline(); showResultsUI();
}

function renderAll() {            // build lại toàn bộ (sau khi đổi tên speaker)
  transcript.innerHTML = "";
  state.turns.forEach((t, i) => transcript.appendChild(turnNode(t, i)));
  renderTimeline();
}

function clearResults() {
  transcript.innerHTML = "";
  for (const k in speakerColor) delete speakerColor[k];
  state.turns = []; state.duration = 0; state.audioName = null;
  if (state.audioURL) { URL.revokeObjectURL(state.audioURL); state.audioURL = null; }
  $("audioEl").removeAttribute("src");
  $("player").classList.add("hidden");
  $("exports").classList.add("hidden");
  $("timeline").classList.add("hidden");
}

function showResultsUI() {
  if (state.turns.length) {
    $("exports").classList.remove("hidden");
    $("timeline").classList.remove("hidden");
  }
}

function escapeHtml(s) { const d = document.createElement("div"); d.textContent = s; return d.innerHTML; }

// ── Đổi tên người nói (relabel tại chỗ, giữ màu) ───────────────────────
function renameSpeaker(old) {
  const name = (prompt("Tên người nói:", old) || "").trim();
  if (!name || name === old) return;
  state.turns.forEach((t) => { if (t.speaker === old) t.speaker = name; });
  if (old in speakerColor && !(name in speakerColor)) {
    speakerColor[name] = speakerColor[old]; delete speakerColor[old];
  }
  renderAll();
}

// ── Audio player đồng bộ (upload mode) ─────────────────────────────────
function seekTo(t) {
  const a = $("audioEl");
  if (a.getAttribute("src")) { a.currentTime = t; a.play().catch(() => {}); }
}
$("audioEl").ontimeupdate = () => {
  const ct = $("audioEl").currentTime;
  state.turns.forEach((t, i) => {
    const on = ct >= t.start && ct < t.end;
    $("turn-" + i)?.classList.toggle("active", on);
    $("seg-" + i)?.classList.toggle("active", on);
  });
};

// ── Timeline người nói ──────────────────────────────────────────────────
function renderTimeline() {
  const tl = $("timeline");
  if (!state.turns.length) { tl.innerHTML = ""; return; }
  const dur = state.duration || 1;
  tl.innerHTML = state.turns.map((t, i) => {
    const left = (t.start / dur * 100).toFixed(2);
    const w = Math.max((t.end - t.start) / dur * 100, 0.4).toFixed(2);
    return `<div class="seg" id="seg-${i}" title="${escapeHtml(t.speaker)} ${fmt(t.start)}→${fmt(t.end)}"
      style="left:${left}%;width:${w}%;background:${colorFor(t.speaker)}"></div>`;
  }).join("");
  tl.querySelectorAll(".seg").forEach((s, i) => { s.onclick = () => seekTo(state.turns[i].start); });
}

// ── Xuất transcript (client-side) ──────────────────────────────────────
const pad = (n, w = 2) => String(n).padStart(w, "0");
function clock(t, sep) {
  const h = Math.floor(t / 3600), m = Math.floor(t % 3600 / 60),
        s = Math.floor(t % 60), ms = Math.round((t % 1) * 1000);
  return `${pad(h)}:${pad(m)}:${pad(s)}${sep}${pad(ms, 3)}`;
}
const csvCell = (s) => `"${String(s).replace(/"/g, '""')}"`;
const EXPORT = {
  txt: () => state.turns.map((t) => `[${fmt(t.start)} → ${fmt(t.end)}] ${t.speaker}: ${t.text}`).join("\n"),
  srt: () => state.turns.map((t, i) => `${i + 1}\n${clock(t.start, ",")} --> ${clock(t.end, ",")}\n${t.speaker}: ${t.text}`).join("\n\n") + "\n",
  vtt: () => "WEBVTT\n\n" + state.turns.map((t) => `${clock(t.start, ".")} --> ${clock(t.end, ".")}\n<v ${t.speaker}>${t.text}`).join("\n\n") + "\n",
  csv: () => "speaker,start,end,text\n" + state.turns.map((t) => `${csvCell(t.speaker)},${t.start},${t.end},${csvCell(t.text)}`).join("\n"),
  json: () => JSON.stringify(state.turns, null, 2),
};
const MIME = { json: "application/json", csv: "text/csv" };
function download(name, content, mime) {
  const blob = new Blob([content], { type: (mime || "text/plain") + ";charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
document.querySelectorAll("#exports button").forEach((b) => {
  b.onclick = () => {
    const f = b.dataset.fmt;
    const base = (state.audioName || "transcript").replace(/\.[^.]+$/, "");
    download(`${base}.${f}`, EXPORT[f](), MIME[f]);
  };
});

// Caption tạm thời — báo "đang nghe" để giảm cảm giác trễ ~6s của streaming
function showInterim() {
  hideInterim();
  const el = document.createElement("div");
  el.id = "interim"; el.className = "turn interim";
  el.innerHTML = `<div class="avatar listening">🎧</div>
    <div class="bubble pending"><div class="text">Đang nghe<span class="dots"><i>.</i><i>.</i><i>.</i></span></div></div>`;
  transcript.appendChild(el);
  transcript.scrollTop = transcript.scrollHeight;
}
function hideInterim() { const e = $("interim"); if (e) e.remove(); }
function setStats(o) {
  $("stats").innerHTML = Object.entries(o)
    .map(([k, v]) => `<span class="chip">${k} <b>${v}</b></span>`).join("");
}
function setStatus(msg, cls) {
  const s = $("status");
  if (!msg) { s.className = "status hidden"; return; }
  s.className = `status ${cls}`;
  s.innerHTML = cls === "loading" ? `<span class="spinner"></span>${msg}` : msg;
}

// ── Tabs ───────────────────────────────────────────────────────────────
document.querySelectorAll(".tab").forEach((tab) => {
  tab.onclick = () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    const live = tab.dataset.mode === "live";
    $("uploadPane").classList.toggle("hidden", live);
    $("livePane").classList.toggle("hidden", !live);
  };
});

// ── Upload (REST /transcribe) ──────────────────────────────────────────
const drop = $("drop"), fileInput = $("file");
let chosenFile = null;
drop.onclick = () => fileInput.click();
fileInput.onchange = () => pick(fileInput.files[0]);
["dragover", "dragenter"].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.add("hover"); }));
["dragleave", "drop"].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.remove("hover"); }));
drop.addEventListener("drop", (ev) => pick(ev.dataTransfer.files[0]));

const MAX_MB = 100;
const OK_EXT = ["wav", "mp3", "m4a", "flac", "ogg", "opus", "webm", "mp4", "avi", "mkv", "mov"];
function pick(f) {
  if (!f) return;
  const ext = (f.name.split(".").pop() || "").toLowerCase();
  if (!OK_EXT.includes(ext)) {
    setStatus(`Định dạng .${ext} không hỗ trợ (chỉ: ${OK_EXT.join(", ")})`, "error");
    chosenFile = null; $("analyze").disabled = true; return;
  }
  if (f.size > MAX_MB * 1e6) {
    setStatus(`File quá lớn (${(f.size / 1e6).toFixed(0)}MB > ${MAX_MB}MB)`, "error");
    chosenFile = null; $("analyze").disabled = true; return;
  }
  chosenFile = f; setStatus("", null);
  $("fileName").textContent = `${f.name} · ${(f.size / 1e6).toFixed(1)} MB`;
  $("analyze").disabled = false;
}

$("analyze").onclick = async () => {
  if (!chosenFile) return;
  const q = new URLSearchParams({ language: $("lang").value || "vi" });
  if ($("numspk").value) q.set("num_speakers", $("numspk").value);
  const fd = new FormData();
  fd.append("file", chosenFile);

  clearResults(); setStats({}); $("analyze").disabled = true;
  setStatus("Đang phân tích…", "loading");
  try {
    const r = await fetch(`/transcribe?${q}`, { method: "POST", body: fd });
    if (!r.ok) {
      let detail; try { detail = (await r.json()).detail; } catch { /* noop */ }
      throw new Error(detail || `HTTP ${r.status}`);
    }
    const j = await r.json();
    setStatus("", null);
    setStats({ "thời lượng": j.duration_s + "s", RTF: j.rtf + "x", turn: j.num_turns });
    state.audioName = chosenFile.name;
    state.audioURL = URL.createObjectURL(chosenFile);
    $("audioEl").src = state.audioURL;
    $("player").classList.remove("hidden");
    j.turns.forEach(addTurn);
    if (!j.turns.length) transcript.innerHTML = `<div class="empty">Không có lời nói.</div>`;
  } catch (e) {
    setStatus("Lỗi: " + e.message, "error");
  } finally {
    $("analyze").disabled = false;
  }
};

// ── Live mic (WebSocket /ws/stream) ────────────────────────────────────
let audioCtx, micStream, processor, ws, recording = false;
const micBtn = $("micBtn");
micBtn.onclick = () => (recording ? stopMic() : startMic());

async function startMic() {
  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
  } catch {
    $("micState").textContent = "Không truy cập được microphone"; return;
  }
  audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
  const src = audioCtx.createMediaStreamSource(micStream);
  processor = audioCtx.createScriptProcessor(4096, 1, 1);
  const analyser = audioCtx.createAnalyser(); analyser.fftSize = 256;
  src.connect(analyser); src.connect(processor); processor.connect(audioCtx.destination);
  drawLevel(analyser);

  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws/stream`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => {
    const cfg = { language: $("lang").value || "vi" };
    if ($("numspk").value) cfg.num_speakers = +$("numspk").value;
    ws.send(JSON.stringify(cfg));
  };
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.event === "turn") addTurn(m);
    else if (m.event === "done") hideInterim();
  };

  processor.onaudioprocess = (e) => {
    if (ws && ws.readyState === WebSocket.OPEN)
      ws.send(new Float32Array(e.inputBuffer.getChannelData(0)).buffer);
  };

  clearResults(); setStats({}); setStatus("", null);
  showInterim();                                       // hiện "đang nghe…" ngay lập tức
  recording = true;
  micBtn.classList.add("recording");
  $("micState").textContent = "Đang ghi… nhấn để dừng";
}

function stopMic() {
  recording = false;
  micBtn.classList.remove("recording");
  $("micState").textContent = "Đang chốt transcript…";
  const it = $("interim");                             // caption đổi sang "đang hoàn tất"
  if (it) it.querySelector(".text").innerHTML = 'Đang hoàn tất<span class="dots"><i>.</i><i>.</i><i>.</i></span>';
  if (processor) processor.disconnect();
  if (micStream) micStream.getTracks().forEach((t) => t.stop());
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send("EOF");
    ws.onclose = () => { hideInterim(); $("micState").textContent = "Nhấn để bắt đầu ghi"; };
  } else { hideInterim(); }
  if (audioCtx) audioCtx.close();
}

function drawLevel(analyser) {
  const cv = $("level"), ctx = cv.getContext("2d");
  const buf = new Uint8Array(analyser.frequencyBinCount);
  (function loop() {
    if (!recording) { ctx.clearRect(0, 0, cv.width, cv.height); return; }
    requestAnimationFrame(loop);
    analyser.getByteFrequencyData(buf);
    ctx.clearRect(0, 0, cv.width, cv.height);
    const bars = 32, w = cv.width / bars;
    for (let i = 0; i < bars; i++) {
      const h = (buf[i * 2] / 255) * cv.height;
      ctx.fillStyle = `rgba(99,102,241,${0.4 + (buf[i * 2] / 255) * 0.6})`;
      ctx.fillRect(i * w, (cv.height - h) / 2, w - 2, h);
    }
  })();
}

// ── Dashboard (poll /health + /metrics) ────────────────────────────────
async function poll() {
  const el = $("health"), txt = $("healthText");
  try {
    const j = await (await fetch("/ready")).json();    // readiness: model + GPU
    if (j.ready) { el.className = "health ok"; txt.textContent = "sẵn sàng · model đã tải"; }
    else if (!j.gpu_ok) { el.className = "health down"; txt.textContent = "GPU lỗi"; }
    else { el.className = "health"; txt.textContent = "đang chờ · model tải khi cần"; }
  } catch {
    el.className = "health down"; txt.textContent = "mất kết nối";
  }
  try {
    const m = await (await fetch("/metrics")).json();
    renderGauges(m);
  } catch {}
}
function gauge(label, value, unit = "", pct = null, color = "var(--accent)") {
  return `<div class="gauge"><div class="label">${label}</div>
    <div class="value">${value}<small>${unit}</small></div>
    ${pct !== null ? `<div class="bar"><span style="width:${Math.min(pct, 100)}%;background:${color}"></span></div>` : ""}
    </div>`;
}
function renderGauges(m) {
  const rtf = m.avg_rtf ?? 0;
  const rtfColor = rtf > 1 ? "var(--bad)" : rtf > 0.7 ? "var(--warn)" : "var(--ok)";
  $("gauges").innerHTML = [
    gauge("Avg RTF", rtf ? rtf.toFixed(3) : "—", "x", rtf ? rtf * 100 : 0, rtfColor),
    gauge("Thiết bị", m.device ?? "—"),
    gauge("Model load", m.model_load_s ?? "—", "s"),
    gauge("REST requests", m.rest_requests ?? 0),
    gauge("WS sessions", m.ws_sessions ?? 0),
    gauge("Turns", m.turns_emitted ?? 0),
    gauge("Audio xử lý", (m.audio_seconds ?? 0).toFixed(0), "s"),
    gauge("Uptime", (m.uptime_s ?? 0).toFixed(0), "s"),
  ].join("");
}
poll();
setInterval(poll, 2500);
