'use strict';

const fs = require('fs');
const path = require('path');
const dotenv = require('dotenv');

/**
 * 욘봇·카카오 스크립트 공통: 상위 레이어 → 하위 레이어 순으로 로드.
 * - 첫 번째로 실제로 존재하는 파일은 override:false (이미 process.env에 있는 값·OS 환경 유지)
 * - 이후 파일은 override:true (앞선 파일에서 넣은 키도 덮어씀)
 * - 마지막 `.env` 는 기존 단일 파일 사용자용 (가장 높은 우선순위)
 *
 * @param {string} yawnbotRoot - `apps/discord-bots/apps/yawnbot` 절대 경로
 * @param {{ includeKakaoLayer?: boolean }} [opts] - 카카오 스크립트만 true
 */
function applyYawnbotDotenvLayers(yawnbotRoot, opts) {
  const discordBotsRoot = path.join(yawnbotRoot, '..', '..');
  const repoRoot = path.join(discordBotsRoot, '..', '..');

  const files = [
    path.join(repoRoot, '.env.karmolab.common'),
    path.join(discordBotsRoot, '.env.discord-bots'),
    path.join(yawnbotRoot, '.env.yawnbot'),
  ];
  if (opts && opts.includeKakaoLayer) {
    files.push(path.join(yawnbotRoot, '.env.yawnbot.kakao'));
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
