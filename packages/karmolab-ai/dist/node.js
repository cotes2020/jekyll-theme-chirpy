"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.resolveAiStudioTextModelId = resolveAiStudioTextModelId;
exports.createAiStudioTextModel = createAiStudioTextModel;
exports.generateAiStudioText = generateAiStudioText;
exports.generateVertexText = generateVertexText;
exports.parseGenerativeSurfaceFromEnv = parseGenerativeSurfaceFromEnv;
exports.generateBlobTextFromEnvWithOptions = generateBlobTextFromEnvWithOptions;
exports.tryCreateGenerativeTextFromEnv = tryCreateGenerativeTextFromEnv;
exports.generativeEnvHint = generativeEnvHint;
exports.generateClaudeCliText = generateClaudeCliText;
exports.resolveAssistantProvider = resolveAssistantProvider;
exports.generateAssistantText = generateAssistantText;
/**
 * Node 전용: AI Studio(`@google/generative-ai`) 또는 Vertex REST(`fetch`)로 텍스트 생성.
 * 브라우저 번들에 포함하지 말 것 — `import 'karmolab-ai/node'`.
 */
const generative_ai_1 = require("@google/generative-ai");
const index_1 = require("./index");
function resolveAiStudioTextModelId(modelFromEnv) {
    const t = modelFromEnv?.trim();
    return t || index_1.DEFAULT_TEXT_MODEL_ID;
}
/** AI Studio API 키 + 선택적 모델 오버라이드로 텍스트용 GenerativeModel */
function createAiStudioTextModel(apiKey, modelId) {
    const genAI = new generative_ai_1.GoogleGenerativeAI(apiKey.trim());
    return genAI.getGenerativeModel({ model: resolveAiStudioTextModelId(modelId) });
}
/** 단일 문자열 프롬프트 → 응답 텍스트 (AI Studio) */
async function generateAiStudioText(opts) {
    const model = createAiStudioTextModel(opts.apiKey, opts.modelId);
    const ro = opts.signal ? { signal: opts.signal } : undefined;
    const res = await model.generateContent(opts.prompt, ro);
    return res.response.text();
}
/** Vertex Publisher `generateContent` (API 키 인증, 브라우저 `gemini.ts`와 동일 REST 형태) */
async function generateVertexText(opts) {
    const model = resolveAiStudioTextModelId(opts.modelId);
    const loc = (opts.location?.trim() || index_1.DEFAULT_VERTEX_LOCATION).trim() || index_1.DEFAULT_VERTEX_LOCATION;
    const url = (0, index_1.buildVertexPublisherModelUrl)({
        projectId: opts.projectId.trim(),
        location: loc,
        modelId: model,
        method: 'generateContent',
        apiKey: opts.apiKey.trim(),
    });
    const body = {
        contents: [{ role: 'user', parts: [{ text: opts.userText }] }],
        generationConfig: { maxOutputTokens: 8192 },
    };
    const sys = opts.systemInstruction?.trim();
    if (sys) {
        body.systemInstruction = { parts: [{ text: sys }] };
    }
    const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: opts.signal,
    });
    const raw = await res.text();
    let data;
    try {
        data = JSON.parse(raw);
    }
    catch {
        throw new Error(`Vertex 응답 파싱 실패 HTTP ${res.status}: ${raw.slice(0, 400)}`);
    }
    if (!res.ok) {
        throw new Error(data.error?.message || data.error?.status || `Vertex HTTP ${res.status}: ${raw.slice(0, 400)}`);
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
function readSurfaceRaw(env) {
    return (env.KARMOLAB_AI_SURFACE?.trim() ||
        env.GEMINI_SURFACE?.trim() ||
        env.GOOGLE_GEN_SURFACE?.trim() ||
        '');
}
/**
 * `vertex` | `vertex_ai` | `gcp_vertex` → Vertex, 그 외·비어 있음 → AI Studio.
 */
function parseGenerativeSurfaceFromEnv(env = process.env) {
    const s = readSurfaceRaw(env).toLowerCase().replace(/-/g, '_');
    if (s === 'vertex' || s === 'vertex_ai' || s === 'gcp_vertex')
        return 'vertex';
    return 'aiStudio';
}
/**
 * 시스템+맥락+질문을 한 문자열로 묶어 보낼 때(AI Studio `generateContent` / Vertex `generateContent` REST).
 * `surface: inherit` 이면 `KARMOLAB_AI_SURFACE` 등과 동일 규칙.
 */
async function generateBlobTextFromEnvWithOptions(env, blobPrompt, options = {}) {
    const surfaceChoice = options.surface ?? 'inherit';
    const surface = surfaceChoice === 'inherit' ? parseGenerativeSurfaceFromEnv(env) : surfaceChoice;
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
        signal: options.signal,
    });
    return { text, surface: 'aiStudio', modelId: effectiveModelId };
}
function tryCreateGenerativeTextFromEnv(env = process.env) {
    const surface = parseGenerativeSurfaceFromEnv(env);
    if (surface === 'vertex') {
        const apiKey = env.VERTEX_API_KEY?.trim();
        const projectId = env.VERTEX_PROJECT_ID?.trim();
        if (!apiKey || !projectId)
            return null;
        const location = env.VERTEX_LOCATION?.trim() || undefined;
        const modelId = env.GEMINI_MODEL?.trim() || null;
        return {
            surface: 'vertex',
            generateFromPrompt(prompt, signal) {
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
    if (!apiKey)
        return null;
    const modelId = env.GEMINI_MODEL?.trim() || null;
    return {
        surface: 'aiStudio',
        generateFromPrompt(prompt, signal) {
            return generateAiStudioText({ apiKey, modelId, prompt, signal });
        },
    };
}
/** `tryCreateGenerativeTextFromEnv`가 `null`일 때 안내용 */
function generativeEnvHint(env = process.env) {
    if (parseGenerativeSurfaceFromEnv(env) === 'vertex') {
        return 'Vertex 모드: .env에 VERTEX_API_KEY, VERTEX_PROJECT_ID 가 필요합니다. (선택: VERTEX_LOCATION, GEMINI_MODEL)';
    }
    return 'AI Studio 모드: .env에 GEMINI_API_KEY 가 필요합니다. (선택: GEMINI_MODEL, 또는 KARMOLAB_AI_SURFACE=vertex 로 전환)';
}
// ─── Claude CLI 프로바이더 ─────────────────────────────────────────────────
const child_process_1 = require("child_process");
/**
 * 로컬에 설치된 `claude` CLI (`claude --print`)로 텍스트 생성.
 * Claude Max 구독으로 인증된 환경에서 API 키 없이 사용 가능.
 *
 * 환경 변수:
 *   CLAUDE_CLI_COMMAND  : CLI 실행 파일 이름 (기본: claude)
 *   CLAUDE_CLI_TIMEOUT_MS : 타임아웃 ms (기본: 60000)
 */
async function generateClaudeCliText(opts) {
    const cmd = process.env.CLAUDE_CLI_COMMAND?.trim() || 'claude';
    const timeout = opts.timeoutMs ?? parseInt(process.env.CLAUDE_CLI_TIMEOUT_MS || '60000', 10);
    return new Promise((resolve, reject) => {
        // cwd가 있으면 에이전트 모드 (파일 접근 허용), 없으면 단순 텍스트 생성
        const args = opts.cwd
            ? ['--print', '--dangerously-skip-permissions']
            : ['--print'];
        // stdin으로 프롬프트 전달 (arg 길이 제한 우회)
        const child = (0, child_process_1.spawn)(cmd, args, {
            stdio: ['pipe', 'pipe', 'pipe'],
            windowsHide: true,
            ...(opts.cwd ? { cwd: opts.cwd } : {}),
        });
        let stdout = '';
        let stderr = '';
        child.stdout.on('data', (data) => { stdout += data.toString(); });
        child.stderr.on('data', (data) => { stderr += data.toString(); });
        const timer = setTimeout(() => {
            child.kill();
            reject(new Error(`Claude CLI 타임아웃 (${timeout}ms)`));
        }, timeout);
        child.on('close', (code) => {
            clearTimeout(timer);
            if (code === 0 && stdout.trim()) {
                resolve(stdout.trim());
            }
            else {
                reject(new Error(`Claude CLI 종료 코드 ${code}: ${stderr.slice(0, 400)}`));
            }
        });
        child.on('error', (err) => {
            clearTimeout(timer);
            reject(new Error(`Claude CLI 실행 실패: ${err.message} (PATH에 '${cmd}'이 있는지 확인)`));
        });
        child.stdin.write(opts.prompt);
        child.stdin.end();
    });
}
function resolveAssistantProvider(env = process.env) {
    const raw = (env.ASSISTANT_AI_PROVIDER ?? '').trim().toLowerCase();
    if (raw === 'claude-cli' || raw === 'claude')
        return 'claude-cli';
    return 'gemini';
}
/**
 * ASSISTANT_AI_PROVIDER 에 따라 Gemini 또는 Claude CLI로 텍스트 생성.
 * assistant-handler, memory-service 등에서 공통으로 사용.
 *
 * claude-cli 프로바이더일 때 env.ASSISTANT_AGENT_REPO_PATH 가 설정돼 있으면
 * 해당 경로를 cwd로 설정해 에이전트 모드(파일 읽기/편집/명령 실행)로 실행.
 */
async function generateAssistantText(env, prompt, opts = {}) {
    const provider = resolveAssistantProvider(env);
    if (provider === 'claude-cli') {
        const cwd = env.ASSISTANT_AGENT_REPO_PATH?.trim() || undefined;
        const text = await generateClaudeCliText({ prompt, timeoutMs: opts.timeoutMs, cwd });
        return { text, provider: 'claude-cli' };
    }
    const { text } = await generateBlobTextFromEnvWithOptions(env, prompt, { surface: 'inherit' });
    return { text, provider: 'gemini' };
}
