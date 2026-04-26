/**
 * GitHub webhook → Discord 채널 라우팅 설정.
 *
 * 파일: data/webhook-routes.json
 *   {
 *     "default": ["채널ID", ...],
 *     "routes": { "owner/repo": ["채널ID", ...] }
 *   }
 *
 * 매칭: payload.repository.full_name 이 routes 에 있으면 그 채널들로,
 *      없으면 default 로 보낸다.
 */
import fs from 'fs';
import path from 'path';
import { PKG_ROOT } from '../paths';

export interface WebhookRoutes {
  default: string[];
  routes: Record<string, string[]>;
}

const ROUTES_PATH = path.join(PKG_ROOT, 'data', 'webhook-routes.json');

let cached: WebhookRoutes | null = null;

function load(): WebhookRoutes {
  if (cached) return cached;
  try {
    const raw = fs.readFileSync(ROUTES_PATH, 'utf-8');
    const parsed = JSON.parse(raw) as Partial<WebhookRoutes>;
    cached = {
      default: Array.isArray(parsed.default) ? parsed.default.filter((s) => typeof s === 'string') : [],
      routes:
        parsed.routes && typeof parsed.routes === 'object'
          ? Object.fromEntries(
              Object.entries(parsed.routes).filter(
                ([, v]) => Array.isArray(v) && v.every((s) => typeof s === 'string'),
              ),
            )
          : {},
    };
  } catch (e: any) {
    console.warn(`[WebhookRoutes] ${ROUTES_PATH} 로드 실패 — 빈 라우팅으로 시작:`, e?.message ?? e);
    cached = { default: [], routes: {} };
  }
  return cached;
}

/** repo full_name 매칭 채널, 없으면 default. */
export function getChannelsForRepo(fullName: string | null | undefined): string[] {
  const r = load();
  if (fullName && r.routes[fullName]?.length) return r.routes[fullName];
  return r.default;
}

/** 어떤 repo에도 묶이지 않는 메시지(예: 봇 시작 인사). */
export function getDefaultChannels(): string[] {
  return load().default;
}

/** 라우팅이 하나도 없는지(설정 미완료 경고용). */
export function hasAnyRoute(): boolean {
  const r = load();
  return r.default.length > 0 || Object.values(r.routes).some((arr) => arr.length > 0);
}
