/**
 * 반드시 다른 앱 모듈보다 먼저 import 되어야 .env 가 process.env 에 반영됨.
 * (그렇지 않으면 voice-connection 등이 모듈 로드 시점에 VOICE_DEBUG 를 읽어 항상 꺼짐)
 *
 * 로드 순서(존재하는 파일만):
 *   1) `packages/karmolab-ai/.env`  (공통 AI 키 — AI_SURFACE, API 키, Vertex 자격증명)
 *   2) `config/yawnbot-defaults.txt` (커밋 기본값)
 *   3) `.env`                        (로컬 오버라이드, 최우선)
 */
import path from 'path';
import { createRequire } from 'node:module';
import { loadKarmoLabAIEnv } from 'karmolab-ai/node';

// 1. 공통 AI 키
loadKarmoLabAIEnv();

const nodeRequire = createRequire(__filename);
const { applyYawnbotDotenvLayers } = nodeRequire(
  path.join(__dirname, '..', '..', 'scripts', 'load-dotenv-layers.cjs'),
) as { applyYawnbotDotenvLayers: (root: string) => void };

const yawnbotRoot = path.join(__dirname, '..', '..');
// 2. yawnbot-defaults.txt + 3. yawnbot .env
applyYawnbotDotenvLayers(yawnbotRoot);
