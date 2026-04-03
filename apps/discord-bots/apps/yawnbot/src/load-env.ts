/**
 * 반드시 다른 앱 모듈보다 먼저 import 되어야 .env 가 process.env 에 반영됨.
 * (그렇지 않으면 voice-connection 등이 모듈 로드 시점에 VOICE_DEBUG 를 읽어 항상 꺼짐)
 */
import path from 'path';
import { config } from 'dotenv';

config({ path: path.join(__dirname, '..', '..', '.env') });
