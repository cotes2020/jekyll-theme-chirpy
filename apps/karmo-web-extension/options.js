/** content.js 의 기본값과 맞출 것 */
const DEFAULT_INGEST_URL = "http://127.0.0.1:17376/ingest";

const input = document.getElementById("ingestUrl");
const btn = document.getElementById("save");
const resetBtn = document.getElementById("resetDefault");
const status = document.getElementById("status");

function show(msg, ok = true) {
  status.textContent = msg;
  status.style.color = ok ? "#0a0" : "#c00";
}

chrome.storage.sync.get({ ingestUrl: DEFAULT_INGEST_URL }, (items) => {
  input.value = items.ingestUrl || DEFAULT_INGEST_URL;
});

function saveUrl(v) {
  try {
    void new URL(v);
  } catch {
    show("올바른 URL 형식이 아닙니다.", false);
    return;
  }
  chrome.storage.sync.set({ ingestUrl: v }, () => {
    if (chrome.runtime.lastError) {
      show(chrome.runtime.lastError.message, false);
      return;
    }
    input.value = v;
    show("저장됨. 치지직 탭은 그대로 두면 다음 채팅부터 새 주소로 전송됩니다.");
  });
}

btn.addEventListener("click", () => {
  const v = input.value.trim();
  if (!v) {
    show("URL을 입력하세요.", false);
    return;
  }
  saveUrl(v);
});

resetBtn.addEventListener("click", () => {
  saveUrl(DEFAULT_INGEST_URL);
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    btn.click();
  }
});
