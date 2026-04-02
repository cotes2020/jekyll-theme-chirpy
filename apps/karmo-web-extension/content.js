/**
 * 사이트별 컨텐츠 스크립트 — 채팅 DOM 등 페이지 데이터를 로컬로 전달.
 * (치지직) 중첩 span을 전부 긁으면 같은 말이 반복·누적되어 보이므로 줄 단위 innerText만 사용.
 */

const DEFAULT_INGEST_URL = "http://127.0.0.1:17376/ingest";

function getIngestUrl() {
  return new Promise((resolve) => {
    chrome.storage.sync.get({ ingestUrl: DEFAULT_INGEST_URL }, (items) => {
      resolve(items.ingestUrl || DEFAULT_INGEST_URL);
    });
  });
}

/** @param {{ author: string, text: string, ts?: number }} payload */
async function forwardToLocal(payload) {
  const url = await getIngestUrl();
  try {
    await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      mode: "cors"
    });
  } catch (e) {
    console.warn("[KarmoWebExtension] ingest 실패 (로컬 앱이 안 떠 있을 수 있음):", e);
  }
}

/** aside > … > div[2] — 치지직 채팅 스크롤 컨테이너 (레이아웃 바뀌면 수정) */
function getChzzkChatListEl() {
  const aside = document.querySelector("section aside") || document.querySelector("aside");
  if (!aside) {
    return null;
  }
  return aside.querySelector(":scope > div:nth-child(3) > div:nth-child(1) > div:nth-child(2)");
}

/**
 * 한 줄(div)의 innerText만 사용 — 중첩 span을 나열하지 않음.
 */
function normalizeBlockText(el) {
  return el.innerText.replace(/\u00a0/g, " ").replace(/[ \t]+\n/g, "\n").replace(/\n{3,}/g, "\n\n").trim();
}

function parseLineEl(line) {
  const raw = normalizeBlockText(line);
  if (!raw) {
    return null;
  }
  const lines = raw
    .split("\n")
    .map((s) => s.trim())
    .filter(Boolean);
  if (lines.length >= 2) {
    return { author: lines[0], text: lines.slice(1).join(" ") };
  }
  const single = lines[0] ?? raw;
  const colon = single.match(/^([^:]+):\s*(.+)$/s);
  if (colon) {
    const a = colon[1].trim();
    const b = colon[2].trim();
    if (a && b) {
      return { author: a, text: b };
    }
  }
  const m = single.match(/^(\S{1,40})\s+(.+)$/s);
  if (m) {
    return { author: m[1], text: m[2].trim() };
  }
  return { author: "chzzk", text: single };
}

/** 동일 줄이 스캔 반복으로 여러 번 나가지 않게 */
const forwardedKeys = new Set();

function keyForPayload(author, text) {
  return `${author}\u0000${text}`;
}

function processLineEl(line) {
  const parsed = parseLineEl(line);
  if (!parsed || !parsed.text) {
    return;
  }
  const key = keyForPayload(parsed.author, parsed.text);
  if (forwardedKeys.has(key)) {
    return;
  }
  forwardedKeys.add(key);
  if (forwardedKeys.size > 3000) {
    forwardedKeys.clear();
  }
  void forwardToLocal({ author: parsed.author, text: parsed.text, ts: Date.now() });
}

let listObserver = null;
/** 같은 DOM 노드에 observer 중복 부착 방지 */
let attachedListEl = null;
let listWatchTimer = 0;

function attachToChatList(listEl) {
  if (listObserver) {
    listObserver.disconnect();
    listObserver = null;
  }
  for (const line of listEl.querySelectorAll(":scope > div")) {
    processLineEl(line);
  }
  listObserver = new MutationObserver((mutations) => {
    for (const m of mutations) {
      for (const n of m.addedNodes) {
        if (n.nodeType !== Node.ELEMENT_NODE) {
          continue;
        }
        const el = /** @type {Element} */ (n);
        if (el.parentElement === listEl && el.matches("div")) {
          processLineEl(el);
        }
      }
    }
  });
  listObserver.observe(listEl, { childList: true });
}

function tryAttachChatObserver() {
  const listEl = getChzzkChatListEl();
  if (!listEl) {
    return;
  }
  if (attachedListEl === listEl) {
    return;
  }
  attachedListEl = listEl;
  attachToChatList(listEl);
}

/** SPA에서 aside가 늦게 붙거나 리스트가 교체되는 경우 */
const bootObserver = new MutationObserver(() => {
  const listEl = getChzzkChatListEl();
  if (listEl && attachedListEl !== listEl) {
    attachedListEl = null;
    tryAttachChatObserver();
  }
});
bootObserver.observe(document.documentElement, { childList: true, subtree: true });

tryAttachChatObserver();

listWatchTimer = window.setInterval(() => {
  if (!getChzzkChatListEl()) {
    return;
  }
  if (!attachedListEl || !document.contains(attachedListEl)) {
    attachedListEl = null;
    tryAttachChatObserver();
  }
}, 2000);
