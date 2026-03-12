const channelInput = document.getElementById("channelInput");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const popoutBtn = document.getElementById("popoutBtn");
const nameColorBtn = document.getElementById("nameColorBtn");
const chatList = document.getElementById("chatList");
const statusText = document.getElementById("statusText");
const modelText = document.getElementById("modelText");
const statsText = document.getElementById("statsText");
const pillPos = document.getElementById("pillPos");
const pillNeu = document.getElementById("pillNeu");
const pillNeg = document.getElementById("pillNeg");
const tmpl = document.getElementById("messageTemplate");

let ws = null;
const MAX_ROWS = 300;
let useSentimentNameColor = true;
let popoutWindow = null;
let popoutPause = false;

function sentimentClass(sent) {
  const s = (sent || "").toLowerCase();
  if (s.includes("pos")) return "positive";
  if (s.includes("neg")) return "negative";
  return "neutral";
}

function tsLabel(ts) {
  const d = new Date((ts || Date.now() / 1000) * 1000);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function sentimentUserColor(sent) {
  const cls = sentimentClass(sent);
  if (cls === "positive") return "#22c55e";
  if (cls === "negative") return "#ef4444";
  return "#f59e0b";
}

function addSystem(text) {
  const row = document.createElement("div");
  row.className = "msg-row system";
  row.textContent = text;
  chatList.appendChild(row);
  trimRows();
  chatList.scrollTop = chatList.scrollHeight;
}

function addMsg(msg) {
  const node = tmpl.content.firstElementChild.cloneNode(true);
  node.querySelector(".msg-time").textContent = tsLabel(msg.timestamp);
  const sentClass = sentimentClass(msg.sentiment);
  const conf = Math.round((msg.confidence || 0) * 100);
  const sent = node.querySelector(".msg-sent");
  sent.textContent = `${msg.sentiment || "Neutral"} | ${conf}%`;
  sent.classList.add(sentClass);

  const userEl = node.querySelector(".msg-user");
  userEl.textContent = `${msg.username}:`;
  if (useSentimentNameColor) {
    userEl.style.color = sentimentUserColor(msg.sentiment);
  } else if (msg.username_color) {
    userEl.style.color = msg.username_color;
  }

  node.querySelector(".msg-text").textContent = `${msg.text}`;
  chatList.appendChild(node);
  trimRows();
  chatList.scrollTop = chatList.scrollHeight;
}

function trimRows() {
  while (chatList.children.length > MAX_ROWS) {
    chatList.removeChild(chatList.firstElementChild);
  }
}

function updateStats(payload) {
  const total = payload.total || 0;
  const rate = Number(payload.messages_per_second || 0).toFixed(1);
  statsText.textContent = `Total ${total} | ${rate} msg/s`;
  const pct = payload.pct || {};
  pillPos.textContent = `Positive ${(pct.Positive || 0).toFixed(1)}%`;
  pillNeu.textContent = `Neutral ${(pct.Neutral || 0).toFixed(1)}%`;
  pillNeg.textContent = `Negative ${(pct.Negative || 0).toFixed(1)}%`;
}

async function startChannel() {
  const channel = channelInput.value.trim().toLowerCase();
  if (!channel) return;
  const res = await fetch("/api/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ channel }),
  });
  const data = await res.json();
  if (!res.ok) {
    addSystem(`Error: ${data.detail || "start failed"}`);
    return;
  }
  const channelLabel = data.channel || channel;
  statusText.textContent = `Running on #${channelLabel}`;
}

async function stopChannel() {
  await fetch("/api/stop", { method: "POST" });
  statusText.textContent = "Stopped";
}

function connectWs() {
  if (popoutPause) return;
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws/messages`);

  ws.onopen = () => {
    statusText.textContent = "Connected";
    addSystem("WebSocket connected");
    ws.send("ping");
  };

  ws.onmessage = (ev) => {
    const payload = JSON.parse(ev.data);
    if (payload.type === "message") addMsg(payload);
    if (payload.type === "system") addSystem(payload.text);
    if (payload.type === "stats") updateStats(payload);
    if (payload.type === "status") {
      if (payload.running) statusText.textContent = `Running on #${payload.channel}`;
      modelText.textContent = payload.model_source || payload.model_dir || "";
    }
  };

  ws.onclose = () => {
    ws = null;
    statusText.textContent = "Disconnected";
    if (!popoutPause) {
      addSystem("WebSocket disconnected. Reconnecting...");
      setTimeout(connectWs, 1200);
    }
  };
}

function closeWsForPopoutPause() {
  if (ws) {
    try {
      ws.close();
    } catch (_) {
      // ignore
    }
  }
}

function setupPopoutMode() {
  const params = new URLSearchParams(window.location.search);
  const isPopout = params.get("popout") === "1";
  if (isPopout) {
    document.body.classList.add("popout");
  }
}

function updateNameColorButton() {
  nameColorBtn.textContent = useSentimentNameColor ? "Name Color: Sentiment" : "Name Color: Twitch";
}

function toggleNameColorMode() {
  useSentimentNameColor = !useSentimentNameColor;
  localStorage.setItem("useSentimentNameColor", useSentimentNameColor ? "1" : "0");
  updateNameColorButton();
}

function openPopout() {
  const url = `${location.origin}${location.pathname}?popout=1`;
  popoutWindow = window.open(url, "twitch_sentiment_popout", "popup=yes,width=520,height=760");
  if (popoutWindow && !document.body.classList.contains("popout")) {
    popoutPause = true;
    closeWsForPopoutPause();
    statusText.textContent = "Paused (popout active)";
    addSystem("Main window paused while popout is active.");
    const watcher = setInterval(() => {
      if (!popoutWindow || popoutWindow.closed) {
        clearInterval(watcher);
        popoutPause = false;
        statusText.textContent = "Reconnecting...";
        addSystem("Popout closed. Resuming main window stream.");
        connectWs();
      }
    }, 1000);
  }
}

function loadPreferences() {
  useSentimentNameColor = localStorage.getItem("useSentimentNameColor") !== "0";
  updateNameColorButton();
}

startBtn.addEventListener("click", startChannel);
stopBtn.addEventListener("click", stopChannel);
popoutBtn.addEventListener("click", openPopout);
nameColorBtn.addEventListener("click", toggleNameColorMode);
channelInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") startChannel();
});

setupPopoutMode();
loadPreferences();
connectWs();
