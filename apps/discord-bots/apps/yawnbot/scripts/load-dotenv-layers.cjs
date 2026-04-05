'use strict';

const fs = require('fs');
const path = require('path');
const dotenv = require('dotenv');

/**
 * 욘봇·카카오 스크립트 공통: 커밋 기본값 → 로컬 `.env` 순.
 * - 첫 로드: override:false (이미 process.env·OS 값 유지)
 * - `.env`: override:true
 *
 * 파일: `config/yawnbot-defaults.txt` → `.env` (yawnbot 루트)
 *
 * @param {string} yawnbotRoot - `apps/discord-bots/apps/yawnbot` 절대 경로
 */
function applyYawnbotDotenvLayers(yawnbotRoot) {
  const files = [
    path.join(yawnbotRoot, 'config', 'yawnbot-defaults.txt'),
    path.join(yawnbotRoot, '.env'),
  ];

  let seen = 0;
  for (const abs of files) {
    if (!fs.existsSync(abs)) continue;
    dotenv.config({ path: abs, override: seen > 0 });
    seen++;
  }
}

module.exports = { applyYawnbotDotenvLayers };
