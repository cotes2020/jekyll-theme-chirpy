/**
 * Node 전용: AI Studio(`@google/generative-ai`) 또는 Vertex REST(`fetch`)로 텍스트 생성.
 * 브라우저 번들에 포함하지 말 것 — `import 'karmolab-ai/node'`.
 */
import { GoogleGenerativeAI } from '@google/generative-ai';
import {
  type GoogleGenerativeSurface,
  buildVertexPublisherModelUrl,
  DEFAULT_VERTEX_LOCATION,
  DEFAULT_TEXT_MODEL_ID,
  getDefaultModelId,
} from './index';

export type { GoogleGenerativeSurface };

export type ChatContent = { role: 'user' | 'model'; parts: [{ text: string }] };

export function resolveAiStudioTextModelId(modelFromEnv?: string | null): string {
  const t = modelFromEnv?.trim();
  return t || DEFAULT_TEXT_MODEL_ID;
}

/** AI Studio API 키 + 선택적 모델 오버라이드로 텍스트용 GenerativeModel */
export function createAiStudioTextModel(apiKey: string, modelId?: string | null) {
  const genAI = new GoogleGenerativeAI(apiKey.trim());
  return genAI.getGenerativeModel({ model: resolveAiStudioTextModelId(modelId) });
}

/** 단일 문자열 프롬프트 → 응답 텍스트 (AI Studio) */
export async function generateAiStudioText(opts: {
  apiKey: string;
  modelId?: string | null;
  prompt: string;
  systemInstruction?: string;
  signal?: AbortSignal;
}): Promise<string> {
  const genAI = new GoogleGenerativeAI(opts.apiKey.trim());
  const model = genAI.getGenerativeModel({
    model: resolveAiStudioTextModelId(opts.modelId),
    ...(opts.systemInstruction ? { systemInstruction: opts.systemInstruction } : {}),
  });
  const ro = opts.signal ? { signal: opts.signal } : undefined;
  const res = await model.generateContent(opts.prompt, ro);
  return res.response.text();
}

/** 멀티턴 대화 히스토리 + 현재 메시지 → 응답 텍스트 (AI Studio Chat) */
async function generateAiStudioChatText(opts: {
  apiKey: string;
  modelId?: string | null;
  systemInstruction?: string;
  history: ChatContent[];
  message: string;
  signal?: AbortSignal;
}): Promise<string> {
  const model = createAiStudioTextModel(opts.apiKey, opts.modelId);
  const chat = model.startChat({
    history: opts.history,
    ...(opts.systemInstruction ? { systemInstruction: opts.systemInstruction } : {}),
  });
  const ro = opts.signal ? { signal: opts.signal } : undefined;
  const res = await chat.sendMessage(opts.message, ro);
  return res.response.text();
}

type VertexJson = {
  error?: { message?: string; status?: string };
  candidates?: Array<{ content?: { parts?: Array<{ text?: string }> } }>;
};

/** Vertex Publisher `generateContent` (API 키 인증, 브라우저 `gemini.ts`와 동일 REST 형태) */
export async function generateVertexText(opts: {
  apiKey: string;
  projectId: string;
  location?: string | null;
  modelId?: string | null;
  userText: string;
  systemInstruction?: string | null;
  safetyThreshold?: string | null;
  signal?: AbortSignal;
}): Promise<string> {
  const model = resolveAiStudioTextModelId(opts.modelId);
  const loc = (opts.location?.trim() || DEFAULT_VERTEX_LOCATION).trim() || DEFAULT_VERTEX_LOCATION;
  const url = buildVertexPublisherModelUrl({
    projectId: opts.projectId.trim(),
    location: loc,
    modelId: model,
    method: 'generateContent',
    apiKey: opts.apiKey.trim(),
  });
  const body: Record<string, unknown> = {
    contents: [{ role: 'user', parts: [{ text: opts.userText }] }],
    generationConfig: { maxOutputTokens: 8192 },
  };
  const sys = opts.systemInstruction?.trim();
  if (sys) {
    body.systemInstruction = { parts: [{ text: sys }] };
  }
  const threshold = opts.safetyThreshold?.trim();
  if (threshold) {
    body.safetySettings = [
      { category: 'HARM_CATEGORY_HARASSMENT', threshold },
      { category: 'HARM_CATEGORY_HATE_SPEECH', threshold },
      { category: 'HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold },
      { category: 'HARM_CATEGORY_DANGEROUS_CONTENT', threshold },
    ];
  }
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal: opts.signal,
  });
  const raw = await res.text();
  let data: VertexJson;
  try {
    data = JSON.parse(raw) as VertexJson;
  } catch {
    throw new Error(`Vertex 응답 파싱 실패 HTTP ${res.status}: ${raw.slice(0, 400)}`);
  }
  if (!res.ok) {
    throw new Error(
      data.error?.message || data.error?.status || `Vertex HTTP ${res.status}: ${raw.slice(0, 400)}`,
    );
  }
  if (data.error) {
    throw new Error(data.error.message || data.error.status || 'Vertex API 오류');
  }
  const text = data.candidates?.[0]?.content?.parts?.[0]?.text;
  if (text == null || text === '') {
    throw new Error('응답에 텍스트가 없습니다: ' + JSON.stringify(data).slice(0, 500));
  }
  return text;
}

function readSurfaceRaw(env: NodeJS.ProcessEnv): string {
  return (
    env.KARMOLAB_AI_SURFACE?.trim() ||
    env.GEMINI_SURFACE?.trim() ||
    env.GOOGLE_GEN_SURFACE?.trim() ||
    ''
  );
}

/**
 * `vertex` | `vertex_ai` | `gcp_vertex` → Vertex, 그 외·비어 있음 → AI Studio.
 */
export function parseGenerativeSurfaceFromEnv(env: NodeJS.ProcessEnv = process.env): GoogleGenerativeSurface {
  const s = readSurfaceRaw(env).toLowerCase().replace(/-/g, '_');
  if (s === 'vertex' || s === 'vertex_ai' || s === 'gcp_vertex') return 'vertex';
  return 'aiStudio';
}

export type GenerativeTextClient = {
  surface: GoogleGenerativeSurface;
  /** 단일 사용자 프롬프트(또는 시스템+사용자를 한 덩어리로 넣은 문자열) */
  generateFromPrompt: (prompt: string, signal?: AbortSignal) => Promise<string>;
};

/**
 * `.env` 기준으로 호출 가능한 텍스트 클라이언트를 만듦. 자격이 없으면 `null`.
 *
 * - **AI Studio (기본):** `GEMINI_API_KEY` 필수, `GEMINI_MODEL` 선택
 * - **Vertex:** `KARMOLAB_AI_SURFACE=vertex`(또는 `GEMINI_SURFACE`) + `VERTEX_API_KEY`, `VERTEX_PROJECT_ID` 필수, `VERTEX_LOCATION`·`GEMINI_MODEL` 선택
 */
/** `/yawn` 슬래시: `.env` 기본 vs `aiStudio` / `vertex` 강제 */
export type GenerativeSurfaceOverride = 'inherit' | 'aiStudio' | 'vertex';

/**
 * 시스템+맥락+질문을 한 문자열로 묶어 보낼 때(AI Studio `generateContent` / Vertex `generateContent` REST).
 * `surface: inherit` 이면 `KARMOLAB_AI_SURFACE` 등과 동일 규칙.
 */
export async function generateBlobTextFromEnvWithOptions(
  env: NodeJS.ProcessEnv,
  blobPrompt: string,
  options: {
    surface?: GenerativeSurfaceOverride;
    modelId?: string | null;
    systemInstruction?: string;
    signal?: AbortSignal;
  } = {},
): Promise<{ text: string; surface: GoogleGenerativeSurface; modelId: string }> {
  const surfaceChoice: GenerativeSurfaceOverride = options.surface ?? 'inherit';
  const surface: GoogleGenerativeSurface =
    surfaceChoice === 'inherit' ? parseGenerativeSurfaceFromEnv(env) : surfaceChoice;

  const modelOverride = options.modelId != null ? String(options.modelId).trim() : '';
  const effectiveModelId = resolveAiStudioTextModelId(modelOverride ? modelOverride : env.GEMINI_MODEL);

  if (surface === 'vertex') {
    const apiKey = env.VERTEX_API_KEY?.trim();
    const projectId = env.VERTEX_PROJECT_ID?.trim();
    if (!apiKey || !projectId) {
      throw new Error('Vertex API: .env에 VERTEX_API_KEY와 VERTEX_PROJECT_ID가 필요합니다.');
    }
    const location = env.VERTEX_LOCATION?.trim() || undefined;
    const text = await generateVertexText({
      apiKey,
      projectId,
      location,
      modelId: effectiveModelId,
      userText: blobPrompt,
      systemInstruction: options.systemInstruction,
      safetyThreshold: env.VERTEX_SAFETY_THRESHOLD?.trim() || null,
      signal: options.signal,
    });
    return { text, surface: 'vertex', modelId: effectiveModelId };
  }

  const apiKey = env.GEMINI_API_KEY?.trim();
  if (!apiKey) {
    throw new Error('AI Studio API: .env에 GEMINI_API_KEY가 필요합니다.');
  }
  const text = await generateAiStudioText({
    apiKey,
    modelId: effectiveModelId,
    prompt: blobPrompt,
    systemInstruction: options.systemInstruction,
    signal: options.signal,
  });
  return { text, surface: 'aiStudio', modelId: effectiveModelId };
}

export function tryCreateGenerativeTextFromEnv(
  env: NodeJS.ProcessEnv = process.env,
): GenerativeTextClient | null {
  const surface = parseGenerativeSurfaceFromEnv(env);
  if (surface === 'vertex') {
    const apiKey = env.VERTEX_API_KEY?.trim();
    const projectId = env.VERTEX_PROJECT_ID?.trim();
    if (!apiKey || !projectId) return null;
    const location = env.VERTEX_LOCATION?.trim() || undefined;
    const modelId = env.GEMINI_MODEL?.trim() || null;
    return {
      surface: 'vertex',
      generateFromPrompt(prompt: string, signal?: AbortSignal) {
        return generateVertexText({
          apiKey,
          projectId,
          location,
          modelId,
          userText: prompt,
          signal,
        });
      },
    };
  }
  const apiKey = env.GEMINI_API_KEY?.trim();
  if (!apiKey) return null;
  const modelId = env.GEMINI_MODEL?.trim() || null;
  return {
    surface: 'aiStudio',
    generateFromPrompt(prompt: string, signal?: AbortSignal) {
      return generateAiStudioText({ apiKey, modelId, prompt, signal });
    },
  };
}

/** `tryCreateGenerativeTextFromEnv`가 `null`일 때 안내용 */
export function generativeEnvHint(env: NodeJS.ProcessEnv = process.env): string {
  if (parseGenerativeSurfaceFromEnv(env) === 'vertex') {
    return 'Vertex 모드: .env에 VERTEX_API_KEY, VERTEX_PROJECT_ID 가 필요합니다. (선택: VERTEX_LOCATION, GEMINI_MODEL)';
  }
  return 'AI Studio 모드: .env에 GEMINI_API_KEY 가 필요합니다. (선택: GEMINI_MODEL, 또는 KARMOLAB_AI_SURFACE=vertex 로 전환)';
}

// ─── Claude CLI 프로바이더 ─────────────────────────────────────────────────

import { spawn } from 'child_process';

/**
 * 로컬에 설치된 `claude` CLI (`claude --print`)로 텍스트 생성.
 * Claude Max 구독으로 인증된 환경에서 API 키 없이 사용 가능.
 *
 * 환경 변수:
 *   CLAUDE_CLI_COMMAND  : CLI 실행 파일 이름 (기본: claude)
 *   CLAUDE_CLI_TIMEOUT_MS : 타임아웃 ms (기본: 60000)
 */
export async function generateClaudeCliText(opts: {
  prompt: string;
  timeoutMs?: number;
  /** 에이전트 모드: cwd 지정 시 파일 읽기/편집/명령 실행 가능 */
  cwd?: string;
}): Promise<string> {
  const cmd = process.env.CLAUDE_CLI_COMMAND?.trim() || 'claude';
  const timeout = opts.timeoutMs ?? parseInt(process.env.CLAUDE_CLI_TIMEOUT_MS || '60000', 10);
  const fixedSessionId = 'yawnbot-assistant';

  const runClaude = (useResume: boolean): Promise<string> => {
    return new Promise<string>((resolve, reject) => {
      // 고정 세션: 항상 같은 세션 이름으로 영구 세션 유지
      // 첫 호출: --continue --name yawnbot-assistant (세션 생성 + 이름 지정)
      // 이후 호출: --resume yawnbot-assistant (이름으로 재개)
      const args = opts.cwd
        ? useResume
          ? ['--print', '--resume', fixedSessionId, '--dangerously-skip-permissions']
          : ['--print', '--continue', '--name', fixedSessionId, '--dangerously-skip-permissions']
        : useResume
          ? ['--print', '--resume', fixedSessionId]
          : ['--print', '--continue', '--name', fixedSessionId];

      const child = spawn(cmd, args, {
        stdio: ['pipe', 'pipe', 'pipe'],
        windowsHide: true,
        ...(opts.cwd ? { cwd: opts.cwd } : {}),
      });

      let stdout = '';
      let stderr = '';

      child.stdout.on('data', (data: Buffer) => { stdout += data.toString(); });
      child.stderr.on('data', (data: Buffer) => { stderr += data.toString(); });

      const timer = setTimeout(() => {
        child.kill();
        reject(new Error(`Claude CLI 타임아웃 (${timeout}ms)`));
      }, timeout);

      child.on('close', (code: number | null) => {
        clearTimeout(timer);
        if (code === 0 && stdout.trim()) {
          resolve(stdout.trim());
        } else {
          reject(new Error(`Claude CLI 종료 코드 ${code}: ${stderr.slice(0, 400)}`));
        }
      });

      child.on('error', (err: Error) => {
        clearTimeout(timer);
        reject(new Error(`Claude CLI 실행 실패: ${err.message} (PATH에 '${cmd}'이 있는지 확인)`));
      });

      child.stdin.write(opts.prompt);
      child.stdin.end();
    });
  };

  // 첫 시도: 기존 세션 재개 (--resume)
  try {
    return await runClaude(true);
  } catch (e) {
    // 세션이 없으면 새로 생성 (--continue)
    const err = e instanceof Error ? e.message : String(e);
    if (err.includes('not found') || err.includes('No session') || err.includes('does not match') || err.includes('not a UUID')) {
      console.log(`[Claude CLI] 기존 세션 없음, 새 세션 생성...`);
      return await runClaude(false);
    }
    throw e;
  }
}

// ─── 통합 프로바이더 (ASSISTANT_AI_PROVIDER 로 선택) ────────────────────────

export type AssistantAiProvider = 'gemini' | 'claude-cli';

export function resolveAssistantProvider(env: NodeJS.ProcessEnv = process.env): AssistantAiProvider {
  const raw = (env.ASSISTANT_AI_PROVIDER ?? '').trim().toLowerCase();
  if (raw === 'claude-cli' || raw === 'claude') return 'claude-cli';
  return 'gemini';
}

/**
 * ASSISTANT_AI_PROVIDER 에 따라 Gemini 또는 Claude CLI로 텍스트 생성.
 * assistant-handler, memory-service 등에서 공통으로 사용.
 *
 * claude-cli 프로바이더일 때 env.ASSISTANT_AGENT_REPO_PATH 가 설정돼 있으면
 * 해당 경로를 cwd로 설정해 에이전트 모드(파일 읽기/편집/명령 실행)로 실행.
 */
export async function generateAssistantText(
  env: NodeJS.ProcessEnv,
  prompt: string,
  opts: { timeoutMs?: number; history?: ChatContent[]; systemInstruction?: string } = {},
): Promise<{ text: string; provider: AssistantAiProvider }> {
  const provider = resolveAssistantProvider(env);

  if (provider === 'claude-cli') {
    const cwd = env.ASSISTANT_AGENT_REPO_PATH?.trim() || undefined;
    const text = await generateClaudeCliText({ prompt, timeoutMs: opts.timeoutMs, cwd });
    return { text, provider: 'claude-cli' };
  }

  if (opts.systemInstruction || (opts.history && opts.history.length > 0)) {
    const surface = parseGenerativeSurfaceFromEnv(env);
    if (surface === 'vertex') {
      // Vertex REST API는 systemInstruction 필드를 지원하므로 직접 전달.
      // history는 미지원 — 턴이 있으면 경고만 남김.
      if (opts.history && opts.history.length > 0) {
        console.warn('[karmolab-ai] Vertex는 chat history를 지원하지 않아 history를 무시합니다.');
      }
      const apiKey = env.VERTEX_API_KEY?.trim();
      const projectId = env.VERTEX_PROJECT_ID?.trim();
      if (!apiKey || !projectId) {
        throw new Error('Vertex API: .env에 VERTEX_API_KEY와 VERTEX_PROJECT_ID가 필요합니다.');
      }
      const text = await generateVertexText({
        apiKey,
        projectId,
        location: env.VERTEX_LOCATION?.trim() || null,
        modelId: resolveAiStudioTextModelId(env.GEMINI_MODEL),
        userText: prompt,
        systemInstruction: opts.systemInstruction,
        safetyThreshold: env.VERTEX_SAFETY_THRESHOLD?.trim() || null,
      });
      return { text, provider: 'gemini' };
    }
    const apiKey = env.GEMINI_API_KEY?.trim();
    if (!apiKey) throw new Error('AI Studio API: .env에 GEMINI_API_KEY가 필요합니다.');
    const modelId = resolveAiStudioTextModelId(env.GEMINI_MODEL);
    const text = await generateAiStudioChatText({
      apiKey,
      modelId,
      systemInstruction: opts.systemInstruction,
      history: opts.history ?? [],
      message: prompt,
    });
    return { text, provider: 'gemini' };
  }

  const { text } = await generateBlobTextFromEnvWithOptions(env, prompt, { surface: 'inherit' });
  return { text, provider: 'gemini' };
}

// ─── Vertex Imagen: 이미지 생성 ────────────────────────────────────────────

export type ImagenAspectRatio = '1:1' | '3:4' | '4:3' | '9:16' | '16:9';
export type ImagenPersonGeneration = 'dont_allow' | 'allow_adult' | 'allow_all';
export type ImagenSafetySetting = 'block_most' | 'block_some' | 'block_few' | 'block_none';

export interface VertexImageResult {
  /** 디코딩된 이미지 바이너리 */
  buffer: Buffer;
  /** "image/png" 등 */
  mimeType: string;
}

/**
 * Vertex Publisher `:predict` 로 Imagen 호출.
 * `MODEL_CATALOG.imagen` 의 ID 중 하나를 `modelId`로 전달 (기본 `imagen-4.0-generate-001`).
 */
export async function generateVertexImage(opts: {
  apiKey: string;
  projectId: string;
  location?: string | null;
  modelId?: string | null;
  prompt: string;
  negativePrompt?: string;
  /** 1~4 */
  sampleCount?: number;
  aspectRatio?: ImagenAspectRatio;
  personGeneration?: ImagenPersonGeneration;
  safetySetting?: ImagenSafetySetting;
  signal?: AbortSignal;
}): Promise<VertexImageResult[]> {
  const modelId = opts.modelId?.trim() || getDefaultModelId('imagen');
  const loc = (opts.location?.trim() || DEFAULT_VERTEX_LOCATION).trim() || DEFAULT_VERTEX_LOCATION;
  const url = buildVertexPublisherModelUrl({
    projectId: opts.projectId.trim(),
    location: loc,
    modelId,
    method: 'predict',
    apiKey: opts.apiKey.trim(),
  });

  const parameters: Record<string, unknown> = {
    sampleCount: Math.max(1, Math.min(4, opts.sampleCount ?? 1)),
  };
  if (opts.aspectRatio) parameters.aspectRatio = opts.aspectRatio;
  if (opts.personGeneration) parameters.personGeneration = opts.personGeneration;
  if (opts.safetySetting) parameters.safetySetting = opts.safetySetting;

  const instance: Record<string, unknown> = { prompt: opts.prompt };
  if (opts.negativePrompt) instance.negativePrompt = opts.negativePrompt;

  const body = { instances: [instance], parameters };

  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal: opts.signal,
  });
  const raw = await res.text();
  let data: {
    error?: { message?: string; status?: string };
    predictions?: Array<{ bytesBase64Encoded?: string; mimeType?: string }>;
  };
  try {
    data = JSON.parse(raw);
  } catch {
    throw new Error(`Imagen 응답 파싱 실패 HTTP ${res.status}: ${raw.slice(0, 400)}`);
  }
  if (!res.ok) {
    throw new Error(
      data.error?.message || data.error?.status || `Imagen HTTP ${res.status}: ${raw.slice(0, 400)}`,
    );
  }
  const preds = data.predictions;
  if (!Array.isArray(preds) || preds.length === 0) {
    throw new Error('Imagen 응답에 이미지가 없습니다: ' + raw.slice(0, 400));
  }
  return preds.map((p) => {
    const b64 = p.bytesBase64Encoded;
    if (!b64) {
      throw new Error('Imagen 응답 prediction 에 bytesBase64Encoded 가 없음');
    }
    return {
      buffer: Buffer.from(b64, 'base64'),
      mimeType: p.mimeType || 'image/png',
    };
  });
}

/**
 * `.env` 기반 이미지 생성. `VERTEX_API_KEY` + `VERTEX_PROJECT_ID` 필수.
 * 모델 우선순위: `options.modelId` > `IMAGE_MODEL_ID` > `MODEL_CATALOG.imagen` 기본값
 */
export async function generateImageFromEnvWithOptions(
  env: NodeJS.ProcessEnv,
  prompt: string,
  options: {
    modelId?: string | null;
    sampleCount?: number;
    aspectRatio?: ImagenAspectRatio;
    negativePrompt?: string;
    personGeneration?: ImagenPersonGeneration;
    safetySetting?: ImagenSafetySetting;
    signal?: AbortSignal;
  } = {},
): Promise<{ images: VertexImageResult[]; modelId: string }> {
  const apiKey = env.VERTEX_API_KEY?.trim();
  const projectId = env.VERTEX_PROJECT_ID?.trim();
  if (!apiKey || !projectId) {
    throw new Error('Imagen(Vertex): .env에 VERTEX_API_KEY와 VERTEX_PROJECT_ID가 필요합니다.');
  }
  const location = env.VERTEX_LOCATION?.trim() || undefined;
  const modelId =
    options.modelId?.trim() || env.IMAGE_MODEL_ID?.trim() || getDefaultModelId('imagen');
  const images = await generateVertexImage({
    apiKey,
    projectId,
    location,
    modelId,
    prompt,
    negativePrompt: options.negativePrompt,
    sampleCount: options.sampleCount,
    aspectRatio: options.aspectRatio,
    personGeneration: options.personGeneration,
    safetySetting: options.safetySetting,
    signal: options.signal,
  });
  return { images, modelId };
}
