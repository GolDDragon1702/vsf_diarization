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

// ── Transcript rendering ───────────────────────────────────────────────
const transcript = $("transcript");
function clearTranscript() { transcript.innerHTML = ""; for (const k in speakerColor) delete speakerColor[k]; }
function addTurn(t) {
  const c = colorFor(t.speaker);
  const pending = t.text === "..." || !t.text;
  const el = document.createElement("div");
  el.className = "turn";
  el.innerHTML = `
    <div class="avatar" style="background:${c}">${initials(t.speaker)}</div>
    <div class="bubble ${pending ? "pending" : ""}" style="border-left-color:${c}">
      <div class="meta"><span style="color:${c}">${t.speaker}</span>
        <span class="time">${fmt(t.start)} → ${fmt(t.end)}</span></div>
      <div class="text">${pending ? "…" : escapeHtml(t.text)}</div>
    </div>`;
  transcript.appendChild(el);
  transcript.scrollTop = transcript.scrollHeight;
}
function escapeHtml(s) { const d = document.createElement("div"); d.textContent = s; return d.innerHTML; }
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
function pick(f) {
  if (!f) return;
  chosenFile = f;
  $("fileName").textContent = `${f.name} · ${(f.size / 1e6).toFixed(1)} MB`;
  $("analyze").disabled = false;
}

$("analyze").onclick = async () => {
  if (!chosenFile) return;
  const q = new URLSearchParams({ language: $("lang").value || "vi" });
  if ($("numspk").value) q.set("num_speakers", $("numspk").value);
  const fd = new FormData();
  fd.append("file", chosenFile);

  clearTranscript(); setStats({}); $("analyze").disabled = true;
  setStatus("Đang phân tích…", "loading");
  try {
    const r = await fetch(`/transcribe?${q}`, { method: "POST", body: fd });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const j = await r.json();
    setStatus("", null);
    setStats({ "thời lượng": j.duration_s + "s", RTF: j.rtf + "x", turn: j.num_turns });
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
  };

  processor.onaudioprocess = (e) => {
    if (ws && ws.readyState === WebSocket.OPEN)
      ws.send(new Float32Array(e.inputBuffer.getChannelData(0)).buffer);
  };

  clearTranscript(); setStats({}); setStatus("", null);
  recording = true;
  micBtn.classList.add("recording");
  $("micState").textContent = "Đang ghi… nhấn để dừng";
}

function stopMic() {
  recording = false;
  micBtn.classList.remove("recording");
  $("micState").textContent = "Đang chốt transcript…";
  if (processor) processor.disconnect();
  if (micStream) micStream.getTracks().forEach((t) => t.stop());
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send("EOF");
    ws.onclose = () => ($("micState").textContent = "Nhấn để bắt đầu ghi");
  }
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
  try {
    const h = await (await fetch("/health")).json();
    const el = $("health");
    el.className = "health ok";
    $("healthText").textContent = h.models_loaded ? "sẵn sàng · model đã tải" : "sẵn sàng · model tải khi cần";
  } catch {
    $("health").className = "health down"; $("healthText").textContent = "mất kết nối";
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
