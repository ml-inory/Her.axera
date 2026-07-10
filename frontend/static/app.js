const DEFAULTS = {
  asr: ["mock_asr", "wenet_onnx", "sensevoice", "fireredasr_aed"],
  llm: ["mock_llm", "deepseek", "openai_compat"],
  tts: ["mock_tts", "edge_tts", "kokoro", "zipvoice"],
  voices: [
    "zh-CN-XiaoxiaoNeural",
    "zh-CN-XiaoyiNeural",
    "zh-CN-YunxiNeural",
    "zh-CN-YunjianNeural",
    "female_default",
    "male_default",
  ],
};

const els = {
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
};

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
  select.replaceChildren(...values.map((value) => new Option(value, value, false, value === preferred)));
}

function initControls() {
  els.apiBase.value = defaultApiBase();
  els.sessionId.value = localStorage.getItem("her.sessionId") || "demo-session";
  fillSelect(els.asrProvider, DEFAULTS.asr, "mock_asr");
  fillSelect(els.llmProvider, DEFAULTS.llm, "mock_llm");
  fillSelect(els.ttsProvider, DEFAULTS.tts, "mock_tts");
  fillSelect(els.voice, DEFAULTS.voices, "zh-CN-XiaoxiaoNeural");
}

async function loadProviders() {
  const base = els.apiBase.value.replace(/\/$/, "");
  try {
    const [asr, llm, tts] = await Promise.all([
      fetch(`${base}/v1/asr/providers`).then((r) => r.json()).catch(() => null),
      fetch(`${base}/v1/llm/providers`).then((r) => r.json()).catch(() => null),
      fetch(`${base}/v1/tts/providers`).then((r) => r.json()).catch(() => null),
    ]);
    if (asr?.providers) fillSelect(els.asrProvider, asr.providers.map((p) => p.name), els.asrProvider.value);
    if (llm?.providers) fillSelect(els.llmProvider, llm.providers.map((p) => p.name), els.llmProvider.value);
    if (tts?.providers) fillSelect(els.ttsProvider, tts.providers.map((p) => p.name), els.ttsProvider.value);
  } catch (error) {
    addEvent("provider_load_failed", error.message);
  }
}

function setConnection(text, mode = "") {
  els.connectionText.textContent = text;
  els.statusDot.className = `dot ${mode}`.trim();
}

function setStatus(text) {
  els.turnStatus.textContent = text;
}

function addEvent(type, detail = "") {
  const item = document.createElement("li");
  item.textContent = detail ? `${type}: ${detail}` : type;
  els.eventLog.prepend(item);
  while (els.eventLog.children.length > 32) {
    els.eventLog.lastElementChild.remove();
  }
}

function addMessage(role, text) {
  const node = els.messageTemplate.content.firstElementChild.cloneNode(true);
  node.classList.add(role);
  node.querySelector(".messageMeta").textContent = role === "user" ? "你" : "Her.axera";
  node.querySelector(".messageText").textContent = text;
  els.conversation.append(node);
  els.conversation.scrollTop = els.conversation.scrollHeight;
  return node;
}

function appendAssistant(text) {
  if (!state.assistantNode) {
    state.assistantNode = addMessage("assistant", "");
  }
  const textNode = state.assistantNode.querySelector(".messageText");
  textNode.textContent = `${textNode.textContent}${text}`;
  els.conversation.scrollTop = els.conversation.scrollHeight;
}

function resetMetrics() {
  state.ttsCount = 0;
  state.ttsMs = 0;
  state.assistantNode = null;
  els.asrMetric.textContent = "-";
  els.llmMetric.textContent = "-";
  els.ttsMetric.textContent = "-";
  els.totalMetric.textContent = "-";
}

function options(extra = {}) {
  const ttsProvider = els.ttsProvider.value;
  return {
    session_id: els.sessionId.value.trim() || "demo-session",
    language: "zh-CN",
    asr_provider: els.asrProvider.value,
    llm_provider: els.llmProvider.value,
    llm_api_key: els.llmApiKey.value.trim() || null,
    tts_provider: ttsProvider,
    voice: els.voice.value,
    output_audio_format: ttsProvider === "edge_tts" ? "mp3" : "wav",
    sample_rate: 24000,
    ...extra,
  };
}

function openSocket() {
  closeSocket();
  const socket = new WebSocket(`${wsBase(els.apiBase.value)}/v1/dialogue/ws`);
  state.socket = socket;
  socket.addEventListener("open", () => setConnection("已连接", "online"));
  socket.addEventListener("message", (event) => handleEvent(JSON.parse(event.data)));
  socket.addEventListener("close", () => setConnection("未连接"));
  socket.addEventListener("error", () => setConnection("连接错误", "error"));
  return socket;
}

function closeSocket() {
  if (state.socket && state.socket.readyState <= WebSocket.OPEN) {
    state.socket.close();
  }
  state.socket = null;
}

function sendWhenOpen(socket, payload) {
  const raw = JSON.stringify(payload);
  if (socket.readyState === WebSocket.OPEN) {
    socket.send(raw);
    return;
  }
  socket.addEventListener("open", () => socket.send(raw), { once: true });
}

function handleEvent(event) {
  addEvent(event.type, event.processing_ms ? `${event.processing_ms} ms` : "");
  if (event.type === "accepted") {
    setStatus("请求已接收");
  } else if (event.type === "speech_started") {
    setStatus("正在采集语音");
  } else if (event.type === "asr_started") {
    setStatus("正在识别");
  } else if (event.type === "asr_partial") {
    setStatus(`识别中：${event.text}`);
  } else if (event.type === "asr") {
    els.asrMetric.textContent = `${event.processing_ms ?? "-"} ms`;
    if (event.text) addMessage("user", event.text);
    setStatus("正在生成回复");
  } else if (event.type === "user_text") {
    setStatus("正在生成回复");
  } else if (event.type === "llm_delta") {
    appendAssistant(event.text || "");
  } else if (event.type === "llm") {
    els.llmMetric.textContent = `${event.processing_ms ?? "-"} ms`;
    if (!state.assistantNode && event.text) appendAssistant(event.text);
  } else if (event.type === "tts_sentence") {
    state.ttsCount += 1;
    state.ttsMs += Number(event.processing_ms || 0);
    els.ttsMetric.textContent = `${state.ttsCount} 句 / ${state.ttsMs} ms`;
    enqueueAudio(event.audio_base64, event.audio_format || "wav");
    setStatus("正在播报");
  } else if (event.type === "done") {
    els.totalMetric.textContent = `${event.total_processing_ms ?? "-"} ms`;
    setStatus("完成");
    closeSocket();
  } else if (event.type === "interrupted") {
    setStatus("已打断");
  } else if (event.type === "error") {
    setConnection("链路错误", "error");
    setStatus(event.error?.message || "请求失败");
  }
}

function base64ToBlob(base64, format) {
  const raw = atob(base64);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i += 1) bytes[i] = raw.charCodeAt(i);
  const type = format === "mp3" ? "audio/mpeg" : "audio/wav";
  return new Blob([bytes], { type });
}

function enqueueAudio(base64, format) {
  if (!base64) return;
  state.audioQueue.push(base64ToBlob(base64, format));
  if (!state.audioPlaying) playNextAudio();
}

function playNextAudio() {
  const blob = state.audioQueue.shift();
  if (!blob) {
    state.audioPlaying = false;
    state.activeAudio = null;
    return;
  }
  state.audioPlaying = true;
  const url = URL.createObjectURL(blob);
  const audio = new Audio(url);
  state.activeAudio = audio;
  audio.addEventListener("ended", () => {
    URL.revokeObjectURL(url);
    playNextAudio();
  });
  audio.addEventListener("error", () => {
    URL.revokeObjectURL(url);
    playNextAudio();
  });
  audio.play().catch((error) => addEvent("audio_play_failed", error.message));
}

function stopAudio() {
  state.audioQueue = [];
  if (state.activeAudio) {
    state.activeAudio.pause();
    state.activeAudio.currentTime = 0;
  }
  state.audioPlaying = false;
  setStatus("播报已停止");
}

function floatToInt16(input) {
  const output = new Int16Array(input.length);
  for (let i = 0; i < input.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, input[i]));
    output[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
  }
  return output;
}

function pcm16ToBase64(pcm) {
  const bytes = new Uint8Array(pcm.buffer);
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  }
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
  sendWhenOpen(socket, {
    type: "speech_start",
    turn_id: turnId,
    input_sample_rate: audioContext.sampleRate,
    channels: 1,
    ...options(),
  });
  processor.onaudioprocess = (event) => {
    if (!state.socket || state.socket.readyState !== WebSocket.OPEN) return;
    const channel = event.inputBuffer.getChannelData(0);
    state.waveform = new Float32Array(channel);
    const pcm = floatToInt16(channel);
    state.socket.send(JSON.stringify({
      type: "audio_chunk",
      turn_id: turnId,
      audio_base64: pcm16ToBase64(pcm),
    }));
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
  recorder.stream.getTracks().forEach((track) => track.stop());
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
  sendWhenOpen(socket, {
    type: "text",
    text,
    turn_id: `turn_${crypto.randomUUID()}`,
    ...options(),
  });
}

function drawWaveform() {
  const canvas = els.waveform;
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#101820";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#58d6c9";
  ctx.lineWidth = 2;
  ctx.beginPath();
  const data = state.waveform;
  for (let i = 0; i < data.length; i += 1) {
    const x = (i / Math.max(1, data.length - 1)) * width;
    const y = height / 2 + data[i] * (height * 0.42);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  requestAnimationFrame(drawWaveform);
}

els.sendButton.addEventListener("click", sendText);
els.textInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") sendText();
});
els.recordButton.addEventListener("click", () => {
  if (state.recorder) stopRecording();
  else startRecording().catch((error) => {
    setConnection("麦克风错误", "error");
    setStatus(error.message);
  });
});
els.stopAudioButton.addEventListener("click", stopAudio);
els.apiBase.addEventListener("change", () => {
  localStorage.setItem("her.apiBase", els.apiBase.value.trim());
  loadProviders();
});
els.sessionId.addEventListener("change", () => {
  localStorage.setItem("her.sessionId", els.sessionId.value.trim());
});

initControls();
loadProviders();
loadModelStatus();
drawWaveform();
setConnection("待机");

// ---- Model Download ----

const modelEls = {
  downloadSection: document.querySelector("#modelDownloadSection"),
  downloadList: document.querySelector("#modelDownloadList"),
  downloadButton: document.querySelector("#downloadModelsButton"),
};

let modelPollTimer = null;

async function loadModelStatus() {
  const base = els.apiBase.value.replace(/\/$/, "");
  try {
    const resp = await fetch(`${base}/v1/models/download/status`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    renderModelStatus(data.models, data.all_ready);
    if (data.all_ready) {
      modelEls.downloadSection.style.display = "none";
      if (modelPollTimer) { clearInterval(modelPollTimer); modelPollTimer = null; }
      return;
    }
    modelEls.downloadSection.style.display = "block";
    // Poll every 2s while downloading
    const hasDownloading = data.models.some(m => m.status === "downloading");
    if (hasDownloading && !modelPollTimer) {
      modelPollTimer = setInterval(loadModelStatus, 2000);
    }
    if (!hasDownloading && modelPollTimer) {
      clearInterval(modelPollTimer);
      modelPollTimer = null;
    }
  } catch (err) {
    // Backend might not have /models endpoint yet — hide panel
    modelEls.downloadSection.style.display = "none";
  }
}

function renderModelStatus(models, allReady) {
  if (!models || models.length === 0) {
    modelEls.downloadSection.style.display = "none";
    return;
  }
  modelEls.downloadSection.style.display = "block";
  modelEls.downloadList.innerHTML = models.map(m => {
    const statusClass = {
      "downloaded": "ready",
      "downloading": "downloading",
      "failed": "failed",
    }[m.status] || "";
    const pct = m.progress_pct || 0;
    return `
      <div class="modelItem">
        <div class="modelLabel">
          <span>${m.display_name}</span>
          <span class="modelStatus ${statusClass}">${statusText(m)}</span>
        </div>
        <div class="progressBar">
          <div class="progressFill ${m.status === 'failed' ? 'failed' : ''}" style="width:${pct}%"></div>
        </div>
        ${m.error_message ? `<div style="font-size:0.75rem;color:var(--danger);margin-top:2px">${m.error_message}</div>` : ""}
      </div>`;
  }).join("");

  modelEls.downloadButton.style.display = allReady ? "none" : "inline-block";
  modelEls.downloadButton.textContent = allReady ? "模型已就绪" : "下载所需模型";
  modelEls.downloadButton.disabled = allReady;
}

function statusText(m) {
  switch (m.status) {
    case "downloaded": return "\u2714 已就绪";
    case "downloading": return `\u23f3 ${Math.round(m.progress_pct)}%`;
    case "failed": return "\u2718 失败";
    case "not_started": return "\u25cb 待下载";
    default: return m.status;
  }
}

async function triggerModelDownload() {
  const base = els.apiBase.value.replace(/\/$/, "");
  modelEls.downloadButton.disabled = true;
  modelEls.downloadButton.textContent = "正在启动...";
  try {
    const resp = await fetch(`${base}/v1/models/download`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_type: null }),
    });
    const data = await resp.json();
    addEvent("model_download", `started ${data.started.length} model(s)`);
    loadModelStatus();
  } catch (err) {
    addEvent("model_download_failed", err.message);
    modelEls.downloadButton.disabled = false;
    modelEls.downloadButton.textContent = "下载所需模型";
  }
}

modelEls.downloadButton.addEventListener("click", triggerModelDownload);
