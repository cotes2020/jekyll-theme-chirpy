/**
 * 반드시 다른 앱 모듈보다 먼저 import 되어야 .env 가 process.env 에 반영됨.
 * (그렇지 않으면 voice-connection 등이 모듈 로드 시점에 VOICE_DEBUG 를 읽어 항상 꺼짐)
 *
 * 로드 순서(존재하는 파일만):
 *   1) 레포 루트 `.env.karmolab.common`
 *   2) `apps/discord-bots/.env.discord-bots`
 *   3) `apps/yawnbot/.env.yawnbot`
 *   4) `apps/yawnbot/.env` (하위 호환 단일 파일)
 *
 * 카카오 스크립트는 추가로 `.env.yawnbot.kakao` 를 읽음 → `scripts/load-dotenv-layers.cjs` 참고.
 */
import path from 'path';
import { createRequire } from 'node:module';

const nodeRequire = createRequire(__filename);
const { applyYawnbotDotenvLayers } = nodeRequire(
  path.join(__dirname, '..', '..', 'scripts', 'load-dotenv-layers.cjs'),
) as { applyYawnbotDotenvLayers: (root: string, opts?: { includeKakaoLayer?: boolean }) => void };

const yawnbotRoot = path.join(__dirname, '..', '..');
applyYawnbotDotenvLayers(yawnbotRoot, { includeKakaoLayer: false });
