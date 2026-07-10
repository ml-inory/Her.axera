// ===== Her.axera Frontend =====
// Onboarding wizard → model download → chat interface

// ---- Config ----
const DEFAULTS = {
  asr: ["mock_asr", "ax_asr", "wenet_onnx", "sensevoice"],
  llm: ["mock_llm", "deepseek", "openai_compat"],
  tts: ["mock_tts", "ax_tts", "edge_tts", "kokoro"],
  voices: ["af_heart", "zf_xiaoxiao", "zh-CN-XiaoxiaoNeural", "female_default", "male_default"],
};

const MODEL_SPECS = [
  { key: "asr_sensevoice", name: "SenseVoice ASR", size: "~120 MB", type: "asr" },
  { key: "asr_whisper_tiny", name: "Whisper Tiny", size: "~80 MB", type: "asr" },
  { key: "tts_kokoro_model", name: "Kokoro TTS Model", size: "~450 MB", type: "tts" },
  { key: "tts_kokoro_voices", name: "Kokoro Voices", size: "~5 MB", type: "tts" },
];

function totalModelSize() {
  let n = 0; MODEL_SPECS.forEach(s => { const m = s.size.match(/(\d+)/); if (m) n += +m[1]; });
  return n >= 1000 ? `${(n / 1000).toFixed(1)} GB` : `${n} MB`;
}

// ---- DOM refs ----
const els = {
  // Onboarding
  onboarding: document.querySelector("#onboarding"),
  obSteps: [...document.querySelectorAll(".obStep")],
  obDots: [...document.querySelectorAll(".obDot")],
  obStartButton: document.querySelector("#obStartButton"),
  obDownloadButton: document.querySelector("#obDownloadButton"),
  obSkipModelsButton: document.querySelector("#obSkipModelsButton"),
  obEnterButton: document.querySelector("#obEnterButton"),
  obModelList: document.querySelector("#obModelList"),
  obDoneText: document.querySelector("#obDoneText"),
  obStatusText: document.querySelector("#obStatusText"),
  obConnectionDot: document.querySelector("#obConnectionDot"),
  // Chat
  chatShell: document.querySelector("#chatShell"),
  apiBase: document.querySelector("#apiBase"),
  sessionId: document.querySelector("#sessionId"),
  asrProvider: document.querySelector("#asrProvider"),
  llmProvider: document.querySelector("#llmProvider"),
  ttsProvider: document.querySelector("#ttsProvider"),
  voice: document.querySelector("#voice"),
  llmApiKey: document.querySelector("#llmApiKey"),
  textInput: document.querySelector("#textInput"),
  sendButton: document.querySelector("#sendButton"),
  recordButton: document.querySelector("#recordButton"),
  stopAudioButton: document.querySelector("#stopAudioButton"),
  turnStatus: document.querySelector("#turnStatus"),
  connectionText: document.querySelector("#connectionText"),
  statusDot: document.querySelector("#statusDot"),
  conversation: document.querySelector("#conversation"),
  eventLog: document.querySelector("#eventLog"),
  waveform: document.querySelector("#waveform"),
  asrMetric: document.querySelector("#asrMetric"),
  llmMetric: document.querySelector("#llmMetric"),
  ttsMetric: document.querySelector("#ttsMetric"),
  totalMetric: document.querySelector("#totalMetric"),
  messageTemplate: document.querySelector("#messageTemplate"),
  // Settings
  settingsDrawer: document.querySelector("#settingsDrawer"),
  openSettingsButton: document.querySelector("#openSettingsButton"),
  closeSettingsButton: document.querySelector("#closeSettingsButton"),
  settingsModelList: document.querySelector("#settingsModelList"),
  settingsDownloadButton: document.querySelector("#settingsDownloadButton"),
};

let obStep = 0;
let modelPollTimer = null;
let modelStates = {};

// ---- State ----
const state = {
  socket: null,
  recorder: null,
  currentTurnId: null,
  assistantNode: null,
  ttsCount: 0,
  ttsMs: 0,
  audioQueue: [],
  audioPlaying: false,
  activeAudio: null,
  waveform: new Float32Array(256),
};

// ============================
//  ONBOARDING WIZARD
// ============================

function setObStep(n) {
  obStep = n;
  els.obSteps.forEach((s, i) => s.classList.toggle("active", i === n));
  els.obDots.forEach((d, i) => {
    d.classList.toggle("active", i === n);
    d.classList.toggle("done", i < n);
  });
}

els.obStartButton.addEventListener("click", () => {
  setObStep(1);
  loadModelStatus();
});

els.obSkipModelsButton.addEventListener("click", () => {
  setObStep(2);
  els.obDoneText.textContent = "使用云端 API，随时可下载本地模型";
  els.obStatusText.textContent = "AX650 · 云端模式";
});

els.obEnterButton.addEventListener("click", enterChat);

async function loadModelStatus() {
  const base = defaultApiBase();
  try {
    const resp = await fetch(`${base}/v1/models/download/status`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    modelStates = {};
    data.models.forEach(m => { modelStates[m.key] = m; });
    renderObModels(data.models);
    updateDownloadButton(data.models);

    const hasDownloading = data.models.some(m => m.status === "downloading");
    if (hasDownloading && !modelPollTimer) {
      modelPollTimer = setInterval(loadModelStatus, 2000);
    }
    if (!hasDownloading && modelPollTimer) {
      clearInterval(modelPollTimer); modelPollTimer = null;
    }
    if (data.all_ready) {
      setObStep(2);
      els.obStatusText.textContent = "AX650 · Python 3.11 · 模型就绪";
      els.obConnectionDot.className = "obConnDot online";
    }
  } catch (err) {
    // Backend unreachable — show error state
    renderObModels([]);
    els.obDownloadButton.textContent = "无法连接后端";
    els.obDownloadButton.disabled = true;
    els.obStatusText.textContent = "AX650 · 后端未连接";
    els.obConnectionDot.className = "obConnDot offline";
  }
}

function renderObModels(models) {
  if (!models.length) {
    els.obModelList.innerHTML = MODEL_SPECS.map(s => `
      <div class="obModelItem">
        <div class="obModelCheck"></div>
        <div class="obModelInfo">
          <div class="obModelName">${s.name}</div>
          <div class="obModelSize">${s.size}</div>
        </div>
        <div class="obModelPct">-</div>
      </div>`).join("");
    return;
  }
  els.obModelList.innerHTML = MODEL_SPECS.map(spec => {
    const state = modelStates[spec.key];
    const done = state && state.status === "downloaded";
    const downloading = state && state.status === "downloading";
    const failed = state && state.status === "failed";
    const pct = state ? state.progress_pct : 0;
    let pctText = "-";
    if (done) pctText = "✓";
    else if (downloading) pctText = `${Math.round(pct)}%`;
    else if (failed) pctText = "✗";
    return `
      <div class="obModelItem">
        <div class="obModelCheck ${done ? "done" : ""} ${downloading ? "downloading" : ""}">${done ? "✓" : ""}</div>
        <div class="obModelInfo">
          <div class="obModelName">${spec.name}</div>
          <div class="obModelSize">${spec.size}</div>
          ${downloading ? `<div class="obModelBar"><div class="obModelBarFill" style="width:${pct}%"></div></div>` : ""}
        </div>
        <div class="obModelPct">${pctText}</div>
      </div>`;
  }).join("");
}

function updateDownloadButton(models) {
  if (!models.length) {
    els.obDownloadButton.textContent = `一键下载 (~${totalModelSize()})`;
    els.obDownloadButton.disabled = false;
    return;
  }
  const allReady = models.every(m => m.status === "downloaded");
  const hasDownloading = models.some(m => m.status === "downloading");
  if (allReady) {
    els.obDownloadButton.textContent = "✓ 全部就绪";
    els.obDownloadButton.disabled = true;
  } else if (hasDownloading) {
    els.obDownloadButton.textContent = "下载中...";
    els.obDownloadButton.disabled = true;
  } else {
    els.obDownloadButton.textContent = `一键下载 (~${totalModelSize()})`;
    els.obDownloadButton.disabled = false;
  }
}

els.obDownloadButton.addEventListener("click", async () => {
  const base = defaultApiBase();
  els.obDownloadButton.disabled = true;
  els.obDownloadButton.textContent = "启动中...";
  try {
    await fetch(`${base}/v1/models/download`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    loadModelStatus();
  } catch (err) {
    els.obDownloadButton.textContent = "重试";
    els.obDownloadButton.disabled = false;
  }
});

// ---- Settings model refresh ----
async function refreshSettingsModels() {
  const base = defaultApiBase();
  try {
    const resp = await fetch(`${base}/v1/models/download/status`);
    if (!resp.ok) return;
    const data = await resp.json();
    if (!data.models.length) return;
    els.settingsModelList.innerHTML = data.models.map(m => {
      const cls = { downloaded: "ready", downloading: "downloading", failed: "failed" }[m.status] || "";
      return `<div class="settingsModelItem"><span>${m.display_name}</span><span class="status ${cls}">${statusText(m)}</span></div>`;
    }).join("");
  } catch (_) {}
}

function statusText(m) {
  if (m.status === "downloaded") return "就绪";
  if (m.status === "downloading") return `${Math.round(m.progress_pct)}%`;
  if (m.status === "failed") return "失败";
  return "待下载";
}

els.settingsDownloadButton.addEventListener("click", async () => {
  const base = defaultApiBase();
  els.settingsDownloadButton.disabled = true;
  try {
    await fetch(`${base}/v1/models/download`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    refreshSettingsModels();
    const interval = setInterval(async () => {
      await refreshSettingsModels();
      try {
        const resp = await fetch(`${base}/v1/models/download/status`);
        const data = await resp.json();
        if (data.all_ready) clearInterval(interval);
      } catch (_) {}
    }, 3000);
  } catch (_) {
    els.settingsDownloadButton.disabled = false;
  }
});

// ============================
//  TRANSITION TO CHAT
// ============================

function enterChat() {
  els.onboarding.style.display = "none";
  els.chatShell.style.display = "grid";
  initControls();
  loadProviders();
  drawWaveform();
  setConnection("待机");
}

// ============================
//  SETTINGS DRAWER
// ============================

els.openSettingsButton.addEventListener("click", () => {
  els.settingsDrawer.classList.add("open");
  refreshSettingsModels();
});
els.closeSettingsButton.addEventListener("click", () => els.settingsDrawer.classList.remove("open"));
document.querySelector(".settingsOverlay").addEventListener("click", () => els.settingsDrawer.classList.remove("open"));

// ============================
//  UTILS
// ============================

function defaultApiBase() {
  const params = new URLSearchParams(location.search);
  const apiFromQuery = params.get("api") || params.get("backend");
  if (apiFromQuery) {
    const normalized = apiFromQuery.replace(/\/$/, "");
    localStorage.setItem("her.apiBase", normalized);
    return normalized;
  }
  const saved = localStorage.getItem("her.apiBase");
  if (saved) return saved;
  if (location.port === "7860") return `${location.protocol}//${location.hostname}:8080`;
  return location.origin;
}

function wsBase(apiBase) {
  const url = new URL(apiBase);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  return url.toString().replace(/\/$/, "");
}

function fillSelect(select, values, preferred) {
  select.replaceChildren(...values.map(v => new Option(v, v, false, v === preferred)));
}

function initControls() {
  els.apiBase.value = defaultApiBase();
  els.sessionId.value = localStorage.getItem("her.sessionId") || "demo-session";
  fillSelect(els.asrProvider, DEFAULTS.asr, "ax_asr");
  fillSelect(els.llmProvider, DEFAULTS.llm, "deepseek");
  fillSelect(els.ttsProvider, DEFAULTS.tts, "ax_tts");
  fillSelect(els.voice, DEFAULTS.voices, "af_heart");
}

async function loadProviders() {
  const base = defaultApiBase();
  try {
    const [asr, llm, tts] = await Promise.all([
      fetch(`${base}/v1/asr/providers`).then(r => r.json()).catch(() => null),
      fetch(`${base}/v1/llm/providers`).then(r => r.json()).catch(() => null),
      fetch(`${base}/v1/tts/providers`).then(r => r.json()).catch(() => null),
    ]);
    if (asr?.providers) fillSelect(els.asrProvider, asr.providers.map(p => p.name), els.asrProvider.value);
    if (llm?.providers) fillSelect(els.llmProvider, llm.providers.map(p => p.name), els.llmProvider.value);
    if (tts?.providers) fillSelect(els.ttsProvider, tts.providers.map(p => p.name), els.ttsProvider.value);
  } catch (_) {}
}

function setConnection(text, mode = "") {
  els.connectionText.textContent = text;
  els.statusDot.className = `dot ${mode}`.trim();
}

function setStatus(text) { els.turnStatus.textContent = text; }

function addEvent(type, detail = "") {
  const item = document.createElement("li");
  item.textContent = detail ? `${type}: ${detail}` : type;
  els.eventLog.prepend(item);
  while (els.eventLog.children.length > 32) els.eventLog.lastElementChild.remove();
}

function addMessage(role, text) {
  const node = els.messageTemplate.content.firstElementChild.cloneNode(true);
  node.classList.add(role);
  const avatar = node.querySelector(".avatar");
  avatar.textContent = role === "user" ? "👤" : "🤖";
  node.querySelector(".messageMeta").textContent = role === "user" ? "你" : "Her.axera";
  node.querySelector(".messageText").textContent = text;
  els.conversation.append(node);
  els.conversation.scrollTop = els.conversation.scrollHeight;
  return node;
}

function appendAssistant(text) {
  if (!state.assistantNode) state.assistantNode = addMessage("assistant", "");
  state.assistantNode.querySelector(".messageText").textContent += text;
  els.conversation.scrollTop = els.conversation.scrollHeight;
}

function resetMetrics() {
  els.asrMetric.textContent = "-";
  els.llmMetric.textContent = "-";
  els.ttsMetric.textContent = "-";
  els.totalMetric.textContent = "-";
  state.ttsCount = 0;
  state.ttsMs = 0;
}

function options() {
  return {
    session_id: els.sessionId.value.trim(),
    asr_provider: els.asrProvider.value,
    llm_provider: els.llmProvider.value,
    tts_provider: els.ttsProvider.value,
    tts_voice: els.voice.value,
    language: "zh",
    llm_api_key: els.llmApiKey.value.trim() || undefined,
  };
}

// ============================
//  WEBSOCKET
// ============================

function openSocket() {
  if (state.socket && state.socket.readyState === WebSocket.OPEN) return state.socket;
  const wsUrl = `${wsBase(defaultApiBase())}/v1/ws/dialogue`;
  const socket = new WebSocket(wsUrl);
  state.socket = socket;
  socket.addEventListener("open", () => setConnection("已连接", "online"));
  socket.addEventListener("close", () => setConnection("未连接", ""));
  socket.addEventListener("error", () => setConnection("连接错误", "error"));
  socket.addEventListener("message", (event) => {
    try {
      handleMessage(JSON.parse(event.data));
    } catch (_) {}
  });
  return socket;
}

function sendWhenOpen(socket, msg) {
  if (socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify(msg));
  else socket.addEventListener("open", () => socket.send(JSON.stringify(msg)), { once: true });
}

function handleMessage(msg) {
  switch (msg.type) {
    case "asr_result":
      els.asrMetric.textContent = `${msg.asr_ms ?? "?"} ms`;
      if (msg.text && !state.assistantNode) appendAssistant(msg.text);
      break;
    case "llm_delta":
      els.llmMetric.textContent = `${msg.llm_ms ?? "?"} ms`;
      appendAssistant(msg.text || "");
      break;
    case "llm_done":
      els.llmMetric.textContent = `${msg.llm_ms ?? "?"} ms`;
      state.assistantNode = null;
      setStatus("播报中");
      break;
    case "tts_segment":
      state.ttsCount++;
      state.ttsMs += msg.tts_ms || 0;
      els.ttsMetric.textContent = `${state.ttsMs} ms`;
      if (msg.audio_base64) {
        const binary = atob(msg.audio_base64);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
        state.audioQueue.push(new Blob([bytes], { type: msg.audio_format === "mp3" ? "audio/mpeg" : "audio/wav" }));
        if (!state.audioPlaying) playNextAudio();
      }
      break;
    case "turn_complete":
      els.totalMetric.textContent = `${msg.total_ms ?? "?"} ms`;
      setConnection("待机");
      setStatus("待机");
      break;
    case "error":
      addEvent("error", msg.message || "未知错误");
      setStatus("错误");
      break;
  }
}

// ============================
//  AUDIO PLAYBACK
// ============================

function playNextAudio() {
  const blob = state.audioQueue.shift();
  if (!blob) { state.audioPlaying = false; state.activeAudio = null; return; }
  state.audioPlaying = true;
  const url = URL.createObjectURL(blob);
  const audio = new Audio(url);
  state.activeAudio = audio;
  audio.addEventListener("ended", () => { URL.revokeObjectURL(url); playNextAudio(); });
  audio.addEventListener("error", () => { URL.revokeObjectURL(url); playNextAudio(); });
  audio.play().catch(err => { addEvent("audio_play_failed", err.message); });
}

function stopAudio() {
  state.audioQueue = [];
  if (state.activeAudio) { state.activeAudio.pause(); state.activeAudio.currentTime = 0; }
  state.audioPlaying = false;
  setStatus("播报已停止");
}

// ============================
//  RECORDING
// ============================

function floatToInt16(input) {
  const output = new Int16Array(input.length);
  for (let i = 0; i < input.length; i++) {
    output[i] = Math.max(-1, Math.min(1, input[i])) * (input[i] < 0 ? 0x8000 : 0x7fff);
  }
  return output;
}

function pcm16ToBase64(pcm) {
  const bytes = new Uint8Array(pcm.buffer);
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(binary);
}

async function startRecording() {
  stopAudio();
  resetMetrics();
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  });
  const audioContext = new AudioContext();
  const source = audioContext.createMediaStreamSource(stream);
  const processor = audioContext.createScriptProcessor(4096, 1, 1);
  const turnId = `turn_${crypto.randomUUID()}`;
  const socket = openSocket();
  state.currentTurnId = turnId;
  sendWhenOpen(socket, { type: "speech_start", turn_id: turnId, input_sample_rate: audioContext.sampleRate, channels: 1, ...options() });
  processor.onaudioprocess = (event) => {
    if (!state.socket || state.socket.readyState !== WebSocket.OPEN) return;
    const channel = event.inputBuffer.getChannelData(0);
    state.waveform = new Float32Array(channel);
    state.socket.send(JSON.stringify({ type: "audio_chunk", turn_id: turnId, audio_base64: pcm16ToBase64(floatToInt16(channel)) }));
  };
  source.connect(processor);
  processor.connect(audioContext.destination);
  state.recorder = { stream, audioContext, source, processor, turnId };
  els.recordButton.classList.add("recording");
  els.recordButton.setAttribute("aria-label", "Stop recording");
  setConnection("录音中", "busy");
  setStatus("正在采集语音");
}

function stopRecording() {
  const recorder = state.recorder;
  if (!recorder) return;
  recorder.processor.disconnect();
  recorder.source.disconnect();
  recorder.stream.getTracks().forEach(t => t.stop());
  recorder.audioContext.close();
  if (state.socket && state.socket.readyState === WebSocket.OPEN) {
    state.socket.send(JSON.stringify({ type: "speech_end", turn_id: recorder.turnId }));
  }
  state.recorder = null;
  els.recordButton.classList.remove("recording");
  els.recordButton.setAttribute("aria-label", "Start recording");
  setStatus("正在提交语音");
}

function sendText() {
  const text = els.textInput.value.trim();
  if (!text) return;
  stopAudio();
  resetMetrics();
  addMessage("user", text);
  els.textInput.value = "";
  const socket = openSocket();
  sendWhenOpen(socket, { type: "text", text, turn_id: `turn_${crypto.randomUUID()}`, ...options() });
}

// ============================
//  WAVEFORM
// ============================

function drawWaveform() {
  const canvas = els.waveform;
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = "#58a6ff";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  const data = state.waveform;
  for (let i = 0; i < data.length; i++) {
    const x = (i / Math.max(1, data.length - 1)) * width;
    const y = height / 2 + data[i] * (height * 0.4);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  requestAnimationFrame(drawWaveform);
}

// ============================
//  EVENT LISTENERS
// ============================

els.sendButton.addEventListener("click", sendText);
els.textInput.addEventListener("keydown", (e) => { if (e.key === "Enter") sendText(); });
els.recordButton.addEventListener("click", () => {
  if (state.recorder) stopRecording();
  else startRecording().catch(err => { setConnection("麦克风错误", "error"); setStatus(err.message); });
});
els.stopAudioButton.addEventListener("click", stopAudio);
els.apiBase.addEventListener("change", () => {
  localStorage.setItem("her.apiBase", els.apiBase.value.trim());
  loadProviders();
});
els.sessionId.addEventListener("change", () => {
  localStorage.setItem("her.sessionId", els.sessionId.value.trim());
});

// ============================
//  INIT
// ============================

// Backend connection check for status bar
(async () => {
  try {
    const resp = await fetch(`${defaultApiBase()}/health`);
    if (resp.ok) {
      els.obStatusText.textContent = "AX650 · Python 3.11 · 后端在线";
      els.obConnectionDot.className = "obConnDot online";
    }
  } catch (_) {}
  els.obDownloadButton.textContent = `一键下载 (~${totalModelSize()})`;
})();
