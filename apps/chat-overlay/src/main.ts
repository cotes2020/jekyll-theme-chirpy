import { app, BrowserWindow, Menu, Tray, nativeImage } from "electron";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { chatFeedKindFromEnv, createChatFeed } from "./chat/createChatFeed.js";
import type { ChatFeedSource } from "./chat/types.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

let mainWindow: BrowserWindow | null = null;
let tray: Tray | null = null;
let activeFeed: ChatFeedSource | null = null;
let unsubscribeFeed: (() => void) | null = null;

function resolveRendererHtml(): string {
  return path.join(__dirname, "renderer", "index.html");
}

function createTray(): void {
  const icon = nativeImage.createFromDataURL(
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
  );
  tray = new Tray(icon);
  tray.setToolTip("chat-overlay");
  tray.setContextMenu(
    Menu.buildFromTemplate([
      {
        label: "종료",
        click: (): void => {
          app.quit();
        }
      }
    ])
  );
}

function attachFeed(win: BrowserWindow): void {
  if (unsubscribeFeed) {
    unsubscribeFeed();
    unsubscribeFeed = null;
  }
  activeFeed?.destroy?.();
  activeFeed = null;

  const kind = chatFeedKindFromEnv();
  activeFeed = createChatFeed(kind);
  unsubscribeFeed = activeFeed.subscribe((line) => {
    win.webContents.send("chat-line", line);
  });
}

function createWindow(): void {
  mainWindow = new BrowserWindow({
    width: 420,
    height: 640,
    x: 40,
    y: 40,
    show: true,
    transparent: true,
    frame: false,
    hasShadow: false,
    alwaysOnTop: true,
    skipTaskbar: true,
    resizable: true,
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false
    }
  });

  mainWindow.setMenuBarVisibility(false);
  mainWindow.setAlwaysOnTop(true, "screen-saver");
  mainWindow.setIgnoreMouseEvents(true);

  void mainWindow.loadFile(resolveRendererHtml());

  mainWindow.webContents.once("did-finish-load", () => {
    if (mainWindow) {
      attachFeed(mainWindow);
    }
  });

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

app.whenReady().then(() => {
  createTray();
  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("before-quit", () => {
  if (unsubscribeFeed) {
    unsubscribeFeed();
    unsubscribeFeed = null;
  }
  activeFeed?.destroy?.();
  activeFeed = null;
  tray?.destroy();
  tray = null;
});
