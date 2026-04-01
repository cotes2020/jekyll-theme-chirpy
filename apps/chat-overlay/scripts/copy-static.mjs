import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(__dirname, "..");
const srcDir = path.join(root, "src", "renderer");
const dstDir = path.join(root, "dist", "renderer");

fs.mkdirSync(dstDir, { recursive: true });
for (const name of ["index.html", "style.css"]) {
  fs.copyFileSync(path.join(srcDir, name), path.join(dstDir, name));
}

const preloadSrc = path.join(root, "src", "preload.cjs");
const preloadDst = path.join(root, "dist", "preload.cjs");
fs.copyFileSync(preloadSrc, preloadDst);
