const POLL_INTERVAL_MS = 850;
let overlayHost;
let overlay;
let lastVersion = 0;
let lastVideo;
let seekTimer;

function videoContext() {
  const video = document.querySelector("video");
  return {
    url: location.href,
    title: document.title.replace(/\s*-\s*YouTube\s*$/, ""),
    timestampSeconds: Number(video?.currentTime || 0),
    durationSeconds: Number(video?.duration || 0)
  };
}

function createOverlay() {
  const player = document.querySelector("#movie_player");
  if (!player) return;
  if (overlayHost?.isConnected) return;

  overlayHost = document.createElement("div");
  overlayHost.id = "cuecast-extension-overlay";
  overlay = overlayHost.attachShadow({ mode: "open" });
  overlay.innerHTML = `
    <style>
      :host{position:absolute;top:58px;right:14px;z-index:9999;pointer-events:none;font-family:Arial,sans-serif}
      .card{width:230px;padding:12px;color:#f5f7fa;background:rgba(17,20,24,.92);border:1px solid rgba(255,255,255,.18);border-radius:12px;box-sizing:border-box;backdrop-filter:blur(10px)}
      .head,.row,.meta{display:flex;align-items:center;justify-content:space-between;gap:8px}.head{font-size:12px}.status{color:#4ade80}.table{position:relative;width:100%;aspect-ratio:2/1;margin:10px 0;overflow:hidden;background:#176b68;border:7px solid #46515c;border-radius:8px;box-sizing:border-box}
      .ball{position:absolute;width:12px;height:12px;margin:-6px;border-radius:50%;border:1px solid #111;transition:left .15s,top .15s}.white{background:#e8f2ff}.yellow{background:#ffd54f}.red{background:#ef5350}.value{font-size:25px;font-weight:600}.track{height:7px;margin:7px 0;background:#303944;border-radius:999px;overflow:hidden}.track span{display:block;height:100%;width:0;background:#42a5f5;transition:width .2s}.meta{color:#aab4be;font-size:10px}
    </style>
    <section class="card">
      <div class="head"><strong>CueCast</strong><span class="status" id="status">● 연결 중</span></div>
      <div class="table"><i class="ball white" id="white"></i><i class="ball yellow" id="yellow"></i><i class="ball red" id="red"></i></div>
      <div class="row"><span>샷 성공률</span><strong class="value" id="probability">--</strong></div>
      <div class="track"><span id="bar"></span></div>
      <div class="meta"><span id="cue">수구 --</span><span id="confidence">신뢰도 --</span></div>
    </section>`;
  player.appendChild(overlayHost);
}

function renderPrediction(latest) {
  if (!latest?.before || !latest?.prediction) return;
  createOverlay();
  if (!overlay) return;
  for (const color of ["white", "yellow", "red"]) {
    const position = latest.before[color];
    const ball = overlay.getElementById(color);
    ball.style.left = `${position[0] * 100}%`;
    ball.style.top = `${position[1] * 100}%`;
  }
  const probability = latest.prediction.successProbability * 100;
  overlay.getElementById("probability").textContent = `${probability.toFixed(1)}%`;
  overlay.getElementById("bar").style.width = `${probability}%`;
  overlay.getElementById("cue").textContent = `수구 ${latest.shooter === "yellow" ? "노란 공" : "흰 공"}`;
  overlay.getElementById("confidence").textContent = `신뢰도 ${Math.round(latest.prediction.confidence.score * 100)}%`;
  overlay.getElementById("status").textContent = "● 자동 분석 중";
}

async function send(message) {
  const response = await chrome.runtime.sendMessage(message);
  if (!response?.ok) throw new Error(response?.error || "CueCast server error");
  return response.data;
}

async function syncCurrentVideo() {
  const { enabled = false, shooter = "white" } = await chrome.storage.local.get(["enabled", "shooter"]);
  if (!enabled) return;
  const context = videoContext();
  if (!context.url.includes("/watch")) return;
  await send({ type: "START_LIVE", ...context, shooter });
}

function bindVideo() {
  const video = document.querySelector("video");
  if (!video || video === lastVideo) return;
  lastVideo = video;
  video.addEventListener("seeked", () => {
    clearTimeout(seekTimer);
    seekTimer = setTimeout(syncCurrentVideo, 350);
  });
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.type === "GET_VIDEO_CONTEXT") {
    sendResponse(videoContext());
  } else if (message.type === "SYNC_VIDEO") {
    syncCurrentVideo().then(() => sendResponse({ ok: true })).catch((error) => sendResponse({ ok: false, error: error.message }));
    return true;
  }
});

setInterval(async () => {
  createOverlay();
  bindVideo();
  try {
    const { enabled = false } = await chrome.storage.local.get("enabled");
    if (!enabled) {
      if (overlay) overlay.getElementById("status").textContent = "● 분석 정지";
      return;
    }
    const latest = await send({ type: "LATEST" });
    if (latest.version > lastVersion) {
      lastVersion = latest.version;
      renderPrediction(latest);
    }
  } catch (error) {
    if (overlay) overlay.getElementById("status").textContent = "● 서버 확인";
  }
}, POLL_INTERVAL_MS);

document.addEventListener("yt-navigate-finish", () => {
  setTimeout(() => {
    createOverlay();
    bindVideo();
    syncCurrentVideo().catch(() => {});
  }, 700);
});

createOverlay();
bindVideo();
