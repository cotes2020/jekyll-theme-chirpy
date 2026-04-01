const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("chatOverlay", {
  onChatLine(cb) {
    const handler = (_evt, line) => {
      cb(line);
    };
    ipcRenderer.on("chat-line", handler);
    return () => {
      ipcRenderer.removeListener("chat-line", handler);
    };
  }
});
