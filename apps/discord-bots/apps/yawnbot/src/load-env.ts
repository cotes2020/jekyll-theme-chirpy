/**
 * 반드시 다른 앱 모듈보다 먼저 import 되어야 .env 가 process.env 에 반영됨.
 * (그렇지 않으면 voice-connection 등이 모듈 로드 시점에 VOICE_DEBUG 를 읽어 항상 꺼짐)
 *
 * 로드 순서(존재하는 파일만):
 *   1) `config/yawnbot-defaults.txt`(커밋 기본값)  2) `.env`(맨 마지막, 최우선)
 */
import path from 'path';
import { createRequire } from 'node:module';

const nodeRequire = createRequire(__filename);
const { applyYawnbotDotenvLayers } = nodeRequire(
  path.join(__dirname, '..', '..', 'scripts', 'load-dotenv-layers.cjs'),
) as { applyYawnbotDotenvLayers: (root: string) => void };

const yawnbotRoot = path.join(__dirname, '..', '..');
applyYawnbotDotenvLayers(yawnbotRoot);
