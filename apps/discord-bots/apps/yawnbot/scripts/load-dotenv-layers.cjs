'use strict';

const fs = require('fs');
const path = require('path');
const dotenv = require('dotenv');

/**
 * 욘봇·카카오 스크립트 공통: 상위 레이어 → 하위 레이어 순으로 로드.
 * - 첫 로드: override:false (이미 process.env·OS 값 유지)
 * - 이후: override:true
 * - 마지막 `.env` 가 가장 높은 우선순위
 *
 * 파일: `config/yawnbot-defaults.txt`(커밋 기본값) → `.karmolab.common.env` → `.discord-bots.env` → `.yawnbot.env` → (카카오만) `.yawnbot.kakao.env` → `.env`
 *
 * @param {string} yawnbotRoot - `apps/discord-bots/apps/yawnbot` 절대 경로
 * @param {{ includeKakaoLayer?: boolean }} [opts]
 */
function applyYawnbotDotenvLayers(yawnbotRoot, opts) {
  const discordBotsRoot = path.join(yawnbotRoot, '..', '..');
  const repoRoot = path.join(discordBotsRoot, '..', '..');

  const files = [
    path.join(yawnbotRoot, 'config', 'yawnbot-defaults.txt'),
    path.join(repoRoot, '.karmolab.common.env'),
    path.join(discordBotsRoot, '.discord-bots.env'),
    path.join(yawnbotRoot, '.yawnbot.env'),
  ];
  if (opts && opts.includeKakaoLayer) {
    files.push(path.join(yawnbotRoot, '.yawnbot.kakao.env'));
  }
  files.push(path.join(yawnbotRoot, '.env'));

  let seen = 0;
  for (const abs of files) {
    if (!fs.existsSync(abs)) continue;
    dotenv.config({ path: abs, override: seen > 0 });
    seen++;
  }
}

module.exports = { applyYawnbotDotenvLayers };
