const API_BASE = "http://127.0.0.1:8765";

chrome.runtime.onInstalled.addListener(() => {
  chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true });
});

async function api(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {})
    }
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || payload.error || `HTTP ${response.status}`);
  }
  return payload;
}

async function handleMessage(message) {
  switch (message.type) {
    case "HEALTH":
      return api("/api/v1/health");
    case "LATEST":
      return api("/api/v1/detection/latest");
    case "LIVE_STATUS":
      return api("/api/v1/youtube/live/status");
    case "START_LIVE":
      return api("/api/v1/youtube/live/start", {
        method: "POST",
        body: JSON.stringify({
          url: message.url,
          timestamp_seconds: message.timestampSeconds || 0,
          shooter: message.shooter || "white"
        })
      });
    case "STOP_LIVE":
      return api("/api/v1/youtube/live/stop", {
        method: "POST",
        body: "{}"
      });
    case "SET_SHOOTER":
      return api("/api/v1/youtube/live/shooter", {
        method: "POST",
        body: JSON.stringify({ shooter: message.shooter })
      });
    case "SHOT_PROBABILITY":
      return api("/api/v1/shot-probability", {
        method: "POST",
        body: JSON.stringify(message.payload)
      });
    case "MATCH_PROBABILITY":
      return api("/api/v1/match-probability", {
        method: "POST",
        body: JSON.stringify(message.payload)
      });
    default:
      throw new Error(`Unknown message: ${message.type}`);
  }
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  handleMessage(message)
    .then((data) => sendResponse({ ok: true, data }))
    .catch((error) => sendResponse({ ok: false, error: error.message }));
  return true;
});
