const IS_EXTENSION = Boolean(globalThis.chrome?.runtime?.id);
const API_BASE = "http://127.0.0.1:8765";
const $ = (id) => document.getElementById(id);
let shooter = "white";
let enabled = false;
let latestVersion = 0;
let currentContext = null;

async function directApi(message) {
  const routes = {
    HEALTH: ["/api/v1/health", "GET"],
    LATEST: ["/api/v1/detection/latest", "GET"],
    LIVE_STATUS: ["/api/v1/youtube/live/status", "GET"],
    START_LIVE: ["/api/v1/youtube/live/start", "POST"],
    STOP_LIVE: ["/api/v1/youtube/live/stop", "POST"],
    SET_SHOOTER: ["/api/v1/youtube/live/shooter", "POST"],
    SHOT_PROBABILITY: ["/api/v1/shot-probability", "POST"],
    MATCH_PROBABILITY: ["/api/v1/match-probability", "POST"]
  };
  const [path, method] = routes[message.type];
  let payload = message.payload;
  if (message.type === "START_LIVE") payload = { url: message.url, timestamp_seconds: message.timestampSeconds || 0, shooter: message.shooter || "white" };
  if (message.type === "SET_SHOOTER") payload = { shooter: message.shooter };
  const response = await fetch(`${API_BASE}${path}`, { method, headers: { "Content-Type": "application/json" }, body: method === "POST" ? JSON.stringify(payload || {}) : undefined });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || data.error);
  return data;
}

async function send(message) {
  if (!IS_EXTENSION) return directApi(message);
  const response = await chrome.runtime.sendMessage(message);
  if (!response?.ok) throw new Error(response?.error || "CueCast server error");
  return response.data;
}

async function getVideoContext() {
  if (!IS_EXTENSION) return { url: "https://www.youtube.com/watch?v=DjRfiNzzwbk", title: "CueCast YouTube 미리보기", timestampSeconds: 15, durationSeconds: 3665 };
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id || !tab.url?.includes("youtube.com/watch")) throw new Error("YouTube 영상 탭을 먼저 열어주세요");
  return chrome.tabs.sendMessage(tab.id, { type: "GET_VIDEO_CONTEXT" });
}

function showTab(name) {
  const live = name === "live";
  $("live-view").hidden = !live;
  $("match-view").hidden = live;
  $("live-tab").setAttribute("aria-selected", String(live));
  $("match-tab").setAttribute("aria-selected", String(!live));
  if (!live) predictMatch();
}

function renderBalls(before) {
  for (const color of ["white", "yellow", "red"]) {
    const ball = document.querySelector(`.ball.${color}`);
    ball.style.left = `${before[color][0] * 100}%`;
    ball.style.top = `${before[color][1] * 100}%`;
    ball.classList.toggle("cueball", color === shooter);
  }
  $("coordinates").textContent = `W (${before.white.map((x) => x.toFixed(3)).join(", ")}) · Y (${before.yellow.map((x) => x.toFixed(3)).join(", ")}) · R (${before.red.map((x) => x.toFixed(3)).join(", ")})`;
}

function renderPrediction(latest) {
  if (!latest?.before || !latest?.prediction) return;
  shooter = latest.shooter || shooter;
  renderBalls(latest.before);
  const probability = latest.prediction.successProbability * 100;
  $("probability").textContent = `${probability.toFixed(1)}%`;
  $("difficulty").textContent = (100 - probability).toFixed(1);
  $("confidence").textContent = `${Math.round(latest.prediction.confidence.score * 100)}%`;
  $("probability-bar").style.width = `${probability}%`;
  $("white").classList.toggle("active", shooter === "white");
  $("yellow").classList.toggle("active", shooter === "yellow");
  if (latest.analysis) $("live-detail").textContent = `자동 검출 ${formatTime(latest.analysis.detectedAtSeconds)} · ${latest.analysis.layoutSource || "layout"}`;
}

function formatTime(value) {
  const seconds = Math.max(0, Math.round(Number(value) || 0));
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const rest = seconds % 60;
  return hours ? `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(rest).padStart(2, "0")}` : `${String(minutes).padStart(2, "0")}:${String(rest).padStart(2, "0")}`;
}

async function startCurrent() {
  try {
    currentContext = await getVideoContext();
    $("video-title").textContent = currentContext.title;
    $("live-state").textContent = "자동 연결 중";
    await send({ type: "START_LIVE", ...currentContext, shooter });
    enabled = true;
    if (IS_EXTENSION) await chrome.storage.local.set({ enabled: true, shooter });
    $("live-toggle").textContent = "자동 분석 중지";
  } catch (error) {
    $("live-state").textContent = "연결 실패";
    $("live-detail").textContent = error.message;
  }
}

async function stopLive() {
  try { await send({ type: "STOP_LIVE" }); } catch {}
  enabled = false;
  if (IS_EXTENSION) await chrome.storage.local.set({ enabled: false });
  $("live-toggle").textContent = "자동 분석 시작";
  $("live-state").textContent = "분석 정지";
}

async function selectShooter(value) {
  shooter = value;
  $("white").classList.toggle("active", value === "white");
  $("yellow").classList.toggle("active", value === "yellow");
  if (IS_EXTENSION) await chrome.storage.local.set({ shooter });
  if (enabled) await send({ type: "SET_SHOOTER", shooter });
}

async function predictMatch() {
  $("favorite").textContent = "계산 중";
  try {
    const data = await send({ type: "MATCH_PROBABILITY", payload: { player_a: $("player-a").value.trim(), player_b: $("player-b").value.trim(), avg_a: Number($("avg-a").value), avg_b: Number($("avg-b").value), sets_to_win: Number($("sets").value) } });
    const pa = data.playerA.winProbability * 100;
    const pb = data.playerB.winProbability * 100;
    $("name-a").textContent = data.playerA.name; $("name-b").textContent = data.playerB.name;
    $("avatar-a").textContent = data.playerA.name[0]; $("avatar-b").textContent = data.playerB.name[0];
    $("win-a").textContent = `${pa.toFixed(1)}%`; $("win-b").textContent = `${pb.toFixed(1)}%`;
    $("set-a").textContent = `세트 승률 ${(data.playerA.setProbability * 100).toFixed(1)}%`; $("set-b").textContent = `세트 승률 ${(data.playerB.setProbability * 100).toFixed(1)}%`;
    $("odds-a").style.width = `${pa}%`; $("odds-b").style.width = `${pb}%`; $("odds-a").textContent = `${pa.toFixed(0)}%`; $("odds-b").textContent = `${pb.toFixed(0)}%`;
    $("face-a").classList.toggle("favorite", pa > pb); $("face-b").classList.toggle("favorite", pb > pa);
    const winner = data.likelyScore.winner === "a" ? data.playerA.name : data.playerB.name;
    $("favorite").textContent = Math.abs(pa - pb) < 1 ? "팽팽한 승부" : `${winner} 우세`;
    $("score").textContent = `가장 가능성 높은 스코어 · ${winner} ${data.likelyScore.winnerSets}-${data.likelyScore.loserSets}`;
    $("model").textContent = data.modelVersion;
  } catch (error) { $("favorite").textContent = error.message; }
}

async function poll() {
  try {
    const [status, latest] = await Promise.all([send({ type: "LIVE_STATUS" }), send({ type: "LATEST" })]);
    if (status.state === "running") $("live-state").textContent = status.lastLayoutSeconds != null ? `자동 · ${formatTime(status.lastLayoutSeconds)} 배치` : `분석 중 · ${formatTime(status.positionSeconds)}`;
    else if (status.state === "connecting") $("live-state").textContent = "자동 연결 중";
    else if (status.state === "error") { $("live-state").textContent = "서버 오류"; $("live-detail").textContent = status.lastError || "분석 오류"; }
    if (latest.version > latestVersion) { latestVersion = latest.version; renderPrediction(latest); }
  } catch (error) { $("live-state").textContent = "서버 연결 필요"; $("live-detail").textContent = "local_probability_server.py를 실행하세요."; }
}

$("live-tab").onclick = () => showTab("live");
$("match-tab").onclick = () => showTab("match");
$("live-toggle").onclick = () => enabled ? stopLive() : startCurrent();
$("sync").onclick = startCurrent;
$("white").onclick = () => selectShooter("white");
$("yellow").onclick = () => selectShooter("yellow");
$("match-predict").onclick = predictMatch;

async function initialize() {
  const previewBalls = { white: [.5583, .9381], yellow: [.2336, .7072], red: [.9643, .3434] };
  renderBalls(previewBalls);
  if (!IS_EXTENSION) {
    $("video-title").textContent = "CueCast 확장 UI 미리보기"; $("live-state").textContent = "미리보기";
    try { const prediction = await send({ type: "SHOT_PROBABILITY", payload: { shooter, before: previewBalls, position_error_mm: 25 } }); renderPrediction({ before: previewBalls, shooter, prediction }); } catch {}
    predictMatch(); poll(); return;
  }
  const stored = await chrome.storage.local.get(["enabled", "shooter"]);
  enabled = stored.enabled !== false;
  shooter = stored.shooter || "white";
  if (enabled) await startCurrent(); else $("live-toggle").textContent = "자동 분석 시작";
  poll();
}

initialize();
setInterval(poll, 1000);
