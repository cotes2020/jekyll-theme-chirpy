/**
 * Cursor local runner — `agent acp` JSON-RPC over stdio, then git summary.
 * Usage: node cli/cursor-local-runner.js --cwd <dir> --prompt "<text>" [--mode agent|ask] [--timeoutMs N]
 * Env: CURSOR_LOCAL_REPO_DIR (optional whitelist), CURSOR_AGENT_COMMAND (default: agent),
 *      CURSOR_MAX_PROMPT_CHARS, CURSOR_DIFF_PREVIEW_CHARS,
 *      CURSOR_GIT_SNAPSHOT=baseline|off (baseline = git stash create -u, 작업 트리 유지)
 *      CURSOR_PROMPT_IDLE_MS — (선택) 마지막 청크 후 침묵 시 완료로 간주(ms). 근본 해결은 아님. 0=비활성(기본)
 *      CURSOR_INTERACTIVE_QUESTIONS=0 — cursor/ask_question 을 부모(stdin) 없이 자동 선택(첫 옵션 등)으로 처리
 *
 * Prints a single JSON object to stdout (logs go to stderr).
 */
/* eslint-disable no-console */
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const { execFile } = require('child_process');
const { promisify } = require('util');
const readline = require('readline');
const execFileAsync = promisify(execFile);

function parseArgs(argv) {
    const out = { cwd: null, prompt: null, mode: 'agent', timeoutMs: null };
    for (let i = 2; i < argv.length; i++) {
        const a = argv[i];
        if (a === '--cwd') out.cwd = argv[++i];
        else if (a === '--prompt') out.prompt = argv[++i];
        else if (a === '--mode') out.mode = argv[++i];
        else if (a === '--timeoutMs') out.timeoutMs = parseInt(argv[++i], 10);
    }
    return out;
}

function isWithinAllowedBase(requestedCwd, allowedBase) {
    if (!allowedBase || !String(allowedBase).trim()) return true;
    const target = path.resolve(requestedCwd);
    const base = path.resolve(allowedBase);
    const rel = path.relative(base, target);
    if (rel === '') return true;
    if (rel.startsWith('..') || path.isAbsolute(rel)) return false;
    return true;
}

function emitJson(obj) {
    process.stdout.write(JSON.stringify(obj));
}

function emitJsonLine(obj) {
    process.stdout.write(JSON.stringify(obj) + '\n');
}

/**
 * Windows에서 `agent.cmd` / `*.bat` 또는 PATH의 `agent`(확장자 없음)를
 * `spawn(경로, ['acp'])`로 직접 실행하면 EINVAL이 날 수 있음 → `shell: true` 사용.
 */
function spawnAgentAcp(agentBin, cwd) {
    const stdio = ['pipe', 'pipe', 'pipe'];
    const env = process.env;
    const opts = { stdio, env, cwd, windowsHide: true };
    if (process.platform !== 'win32') {
        return spawn(agentBin, ['acp'], opts);
    }
    const ext = path.extname(agentBin).toLowerCase();
    const needsShell = ext === '' || ext === '.cmd' || ext === '.bat';
    return spawn(agentBin, ['acp'], needsShell ? { ...opts, shell: true } : opts);
}

function runGitSummary(cwd) {
    return (async () => {
        try {
            await execFileAsync('git', ['-C', cwd, 'rev-parse', '--is-inside-work-tree'], {
                encoding: 'utf8',
            });
        } catch {
            return {
                isRepo: false,
                reason: 'not a git repository',
                statusPorcelain: '',
                diffStat: '',
                diffPreview: '',
            };
        }
        const [statusPorcelain, diffStat, diffFull] = await Promise.all([
            execFileAsync('git', ['-C', cwd, 'status', '--porcelain'], { encoding: 'utf8' }).then(
                r => r.stdout,
                () => '',
            ),
            execFileAsync('git', ['-C', cwd, 'diff', '--stat'], { encoding: 'utf8' }).then(
                r => r.stdout,
                () => '',
            ),
            execFileAsync('git', ['-C', cwd, 'diff'], { encoding: 'utf8' }).then(r => r.stdout, () => ''),
        ]);
        const maxPreview = parseInt(process.env.CURSOR_DIFF_PREVIEW_CHARS || '8000', 10);
        const diffPreview =
            diffFull.length > maxPreview ? diffFull.slice(0, maxPreview) + '\n…(truncated)' : diffFull;
        return {
            isRepo: true,
            statusPorcelain: statusPorcelain.trimEnd(),
            diffStat: diffStat.trimEnd(),
            diffPreview: diffPreview.trimEnd(),
        };
    })();
}

async function isGitRepo(cwd) {
    try {
        await execFileAsync('git', ['-C', cwd, 'rev-parse', '--is-inside-work-tree'], { encoding: 'utf8' });
        return true;
    } catch {
        return false;
    }
}

/**
 * 작업 트리를 건드리지 않고 스냅샷 커밋만 만듭니다 (`git stash create -u`).
 * 변경이 없으면 `HEAD`를 baseline으로 씁니다.
 * 이후 `git diff <baseline>`으로 “이번 실행에서만” 바뀐 내용을 계산합니다.
 */
async function createWorkingTreeBaseline(cwd) {
    const raw = String(process.env.CURSOR_GIT_SNAPSHOT || 'baseline').toLowerCase().trim();
    if (raw === 'off' || raw === 'false' || raw === '0') {
        return { mode: 'off', ref: null };
    }
    // 예전 이름: stash push/pop — 위험하므로 baseline(=create)로만 동작
    if (raw === 'stash' || raw === 'baseline' || raw === 'create' || raw === 'safe') {
        if (!(await isGitRepo(cwd))) return { mode: 'no-git', ref: null };

        try {
            const r = await execFileAsync('git', ['-C', cwd, 'stash', 'create', '-u'], { encoding: 'utf8' });
            const hash = (r.stdout || '').trim();
            if (hash) {
                return { mode: 'baseline', ref: hash, method: 'stash-create' };
            }
            const head = await execFileAsync('git', ['-C', cwd, 'rev-parse', 'HEAD'], { encoding: 'utf8' });
            return { mode: 'baseline', ref: (head.stdout || '').trim(), method: 'head' };
        } catch (e) {
            return { mode: 'error', ref: null, error: e.message };
        }
    }
    return { mode: 'off', ref: null, reason: `unknown CURSOR_GIT_SNAPSHOT=${raw}` };
}

/**
 * `git diff <baseRef>` 기준 요약. baseRef가 없으면 기존 `runGitSummary`와 동일(전체 작업 트리).
 */
function runGitSummarySince(cwd, baseRef) {
    return (async () => {
        try {
            await execFileAsync('git', ['-C', cwd, 'rev-parse', '--is-inside-work-tree'], {
                encoding: 'utf8',
            });
        } catch {
            return {
                isRepo: false,
                reason: 'not a git repository',
                statusPorcelain: '',
                diffStat: '',
                diffPreview: '',
            };
        }

        if (!baseRef) {
            return await runGitSummary(cwd);
        }

        const [statusPorcelain, diffStat, diffFull] = await Promise.all([
            execFileAsync('git', ['-C', cwd, 'diff', '--name-status', baseRef], { encoding: 'utf8' }).then(
                r => r.stdout,
                () => '',
            ),
            execFileAsync('git', ['-C', cwd, 'diff', '--stat', baseRef], { encoding: 'utf8' }).then(
                r => r.stdout,
                () => '',
            ),
            execFileAsync('git', ['-C', cwd, 'diff', baseRef], { encoding: 'utf8' }).then(r => r.stdout, () => ''),
        ]);
        const maxPreview = parseInt(process.env.CURSOR_DIFF_PREVIEW_CHARS || '8000', 10);
        const diffPreview =
            diffFull.length > maxPreview ? diffFull.slice(0, maxPreview) + '\n…(truncated)' : diffFull;
        return {
            isRepo: true,
            statusPorcelain: statusPorcelain.trimEnd(),
            diffStat: diffStat.trimEnd(),
            diffPreview: diffPreview.trimEnd(),
            baselineRef: baseRef,
        };
    })();
}

/**
 * @param {object} opts
 * @param {string} opts.cwd
 * @param {string} opts.prompt
 * @param {string} opts.mode
 * @param {number} opts.timeoutMs
 */
async function runAcp(opts) {
    const agentBin = process.env.CURSOR_AGENT_COMMAND || 'agent';
    const child = spawnAgentAcp(agentBin, opts.cwd);

    let nextId = 1;
    const pending = new Map();
    const collectedText = [];
    /** 완료 JSON·디스코드 임베드용: 에이전트→클라이언트 RPC 호출 횟수 */
    let askQuestionRpcCount = 0;
    let createPlanRpcCount = 0;

    function takePending(id) {
        const candidates = [id];
        if (typeof id === 'string' && /^-?\d+$/.test(id)) candidates.push(Number(id));
        if (typeof id === 'number' && Number.isFinite(id)) candidates.push(String(id));
        for (const k of candidates) {
            if (pending.has(k)) {
                const w = pending.get(k);
                pending.delete(k);
                return w;
            }
        }
        return null;
    }

    function beginSend(method, params) {
        const id = nextId++;
        const promise = new Promise((resolve, reject) => {
            pending.set(id, { resolve, reject });
        });
        child.stdin.write(JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n');
        return { id, promise };
    }

    async function send(method, params) {
        const { promise } = beginSend(method, params);
        return promise;
    }

    function respondRequest(requestId, result) {
        child.stdin.write(JSON.stringify({ jsonrpc: '2.0', id: requestId, result }) + '\n');
    }

    const spawnError = new Promise((_, reject) => {
        child.once('error', err => reject(err));
    });

    /**
     * (선택) 마지막 스트림 청크 이후 이 ms 동안 session/prompt JSON-RPC가 없으면 완료로 간주.
     * 기본 120000(2분): 짧은 간격(8초 등)은 긴 추론 구간에서 오탐이 나기 쉬움. 0=끔(진단용).
     */
    let promptTurnId = null;
    let idleTimer = null;
    const idleMs = parseInt(process.env.CURSOR_PROMPT_IDLE_MS || '120000', 10);

    function bumpIdle() {
        if (promptTurnId == null || idleMs <= 0) return;
        clearTimeout(idleTimer);
        idleTimer = setTimeout(() => {
            if (promptTurnId != null && pending.has(promptTurnId)) {
                const waiter = pending.get(promptTurnId);
                pending.delete(promptTurnId);
                promptTurnId = null;
                console.error(
                    '[cursor-local-runner] session/prompt: JSON-RPC 응답 없음 → idle 완료 처리 (CURSOR_PROMPT_IDLE_MS=' +
                        idleMs +
                        'ms)',
                );
                waiter.resolve({ stopReason: 'idle_inferred' });
            }
        }, idleMs);
    }

    function rpcIdMatchesPrompt(msgId) {
        if (promptTurnId == null) return false;
        return msgId === promptTurnId || String(msgId) === String(promptTurnId);
    }

    /**
     * Cursor가 클라이언트로 보내는 JSON-RPC 요청(cursor/…)에 대한 최소 응답 (비대화형 폴백).
     * @see https://cursor.com/docs/cli/acp (Cursor extension methods)
     */
    function defaultCursorExtensionResult(method, params) {
        const p = params && typeof params === 'object' ? params : {};
        switch (method) {
            case 'cursor/ask_question': {
                const choices = p.choices ?? p.options;
                if (Array.isArray(choices) && choices.length > 0) {
                    return { selectedIndex: 0 };
                }
                return { cancelled: true };
            }
            case 'cursor/create_plan':
                return { approved: true };
            case 'cursor/update_todos':
            case 'cursor/task':
            case 'cursor/generate_image':
                return { acknowledged: true };
            default:
                return { acknowledged: true };
        }
    }

    let stdinRl = null;
    /** 부모(디스코드 봇)가 한 줄 JSON을 쓰지 않으면 영구 대기하므로 타임아웃으로 해제. 0=무제한 */
    const questionStdinMs = parseInt(process.env.CURSOR_QUESTION_STDIN_MS || '600000', 10);
    function readStdinAnswerLine() {
        return new Promise((resolve, reject) => {
            if (!process.stdin.readable) {
                reject(new Error('stdin not readable'));
                return;
            }
            if (!stdinRl) {
                stdinRl = readline.createInterface({ input: process.stdin, terminal: false });
            }
            let tid = null;
            const onLine = line => {
                if (tid) clearTimeout(tid);
                stdinRl.off('line', onLine);
                try {
                    resolve(JSON.parse(line));
                } catch (e) {
                    reject(e);
                }
            };
            stdinRl.on('line', onLine);
            if (questionStdinMs > 0) {
                tid = setTimeout(() => {
                    stdinRl.off('line', onLine);
                    reject(
                        new Error(
                            `stdin answer timeout after ${questionStdinMs}ms (CURSOR_QUESTION_STDIN_MS)`,
                        ),
                    );
                }, questionStdinMs);
            }
        });
    }

    /** ask_question 은 순차 처리 (여러 개가 줄을 서서 올 수 있음) */
    let askQuestionChain = Promise.resolve();
    function handleAskQuestionInteractive(msg) {
        const interactiveOff = String(process.env.CURSOR_INTERACTIVE_QUESTIONS || '').toLowerCase() === '0';
        if (interactiveOff) {
            respondRequest(msg.id, defaultCursorExtensionResult('cursor/ask_question', msg.params));
            return;
        }
        // 터미널에서 직접 실행(stdin=TTY)이면 부모 파이프가 없음 → 자동 폴백
        if (process.stdin.isTTY) {
            respondRequest(msg.id, defaultCursorExtensionResult('cursor/ask_question', msg.params));
            return;
        }

        askQuestionChain = askQuestionChain
            .then(async () => {
                emitJsonLine({
                    type: 'cursor_question',
                    rpcId: msg.id,
                    params: msg.params,
                });
                try {
                    const answer = await readStdinAnswerLine();
                    const rid = answer.rpcId;
                    if (rid != null && rid !== msg.id && String(rid) !== String(msg.id)) {
                        console.error('[cursor-local-runner] stdin 응답 rpcId 불일치:', rid, '기대:', msg.id);
                    }
                    if (answer.cancelled === true) {
                        respondRequest(msg.id, { cancelled: true });
                        return;
                    }
                    const idx = answer.selectedIndex;
                    const n = typeof idx === 'number' ? idx : parseInt(String(idx), 10);
                    if (Number.isNaN(n)) {
                        respondRequest(msg.id, { cancelled: true });
                        return;
                    }
                    respondRequest(msg.id, { selectedIndex: n });
                } catch (e) {
                    console.error(
                        '[cursor-local-runner] ask_question stdin 처리 실패:',
                        e.message || e,
                    );
                    respondRequest(msg.id, { cancelled: true });
                    bumpIdle();
                }
            })
            .catch(e => console.error('[cursor-local-runner] ask_question queue:', e.message || e));
    }

    function handleAgentToClientRequest(msg) {
        if (msg.method === 'session/request_permission') {
            respondRequest(msg.id, { outcome: { outcome: 'selected', optionId: 'allow-once' } });
            return;
        }
        if (msg.method === 'cursor/ask_question') {
            askQuestionRpcCount++;
            handleAskQuestionInteractive(msg);
            return;
        }
        if (msg.method === 'cursor/create_plan') {
            createPlanRpcCount++;
        }
        if (typeof msg.method === 'string' && msg.method.startsWith('cursor/')) {
            console.error('[cursor-local-runner] 에이전트→클라이언트 요청', msg.method, '→ 자동 result');
            respondRequest(msg.id, defaultCursorExtensionResult(msg.method, msg.params));
            return;
        }
        console.error('[cursor-local-runner] 미처리 에이전트→클라이언트 요청:', msg.method, JSON.stringify(msg).slice(0, 400));
        respondRequest(msg.id, { acknowledged: true });
    }

    /** readline은 마지막 줄에 \\n이 없으면 영원히 안 올라옴 → 버퍼로 NDJSON 파싱 */
    let stdoutBuf = '';
    function processAcpLine(line) {
        let msg;
        try {
            msg = JSON.parse(line);
        } catch {
            console.error('[cursor-local-runner] bad JSON line:', line.slice(0, 200));
            return;
        }

        if (msg.id != null && (msg.result !== undefined || msg.error !== undefined)) {
            if (rpcIdMatchesPrompt(msg.id)) {
                clearTimeout(idleTimer);
                promptTurnId = null;
            }
            const waiter = takePending(msg.id);
            if (!waiter) {
                console.error('[cursor-local-runner] JSON-RPC 응답 id를 pending에서 찾지 못함:', msg.id);
                return;
            }
            if (msg.error) waiter.reject(new Error(msg.error.message || JSON.stringify(msg.error)));
            else waiter.resolve(msg.result);
            return;
        }

        if (msg.method === 'session/update') {
            const u = msg.params?.update;
            if (u?.sessionUpdate === 'agent_message_chunk' && u.content?.text) {
                collectedText.push(u.content.text);
                emitJsonLine({ type: 'assistant_chunk', text: u.content.text });
                bumpIdle();
            }
            return;
        }

        // 에이전트가 클라이언트로 보내는 JSON-RPC 요청 (응답 필요): session/request_permission, cursor/* …
        if (
            msg.method &&
            msg.id != null &&
            msg.result === undefined &&
            msg.error === undefined
        ) {
            handleAgentToClientRequest(msg);
        }
    }

    function onStdoutData(chunk) {
        stdoutBuf += chunk.toString('utf8');
        let nl;
        while ((nl = stdoutBuf.indexOf('\n')) !== -1) {
            const line = stdoutBuf.slice(0, nl).replace(/\r$/, '');
            stdoutBuf = stdoutBuf.slice(nl + 1);
            if (line.length) processAcpLine(line);
        }
    }

    child.stdout.on('data', onStdoutData);
    child.stdout.on('end', () => {
        const rest = stdoutBuf.trim();
        stdoutBuf = '';
        if (rest) processAcpLine(rest);
    });

    const stderrChunks = [];
    child.stderr.on('data', chunk => {
        stderrChunks.push(chunk.toString());
    });

    const timeoutMs = opts.timeoutMs || parseInt(process.env.CURSOR_TIMEOUT_MS || '600000', 10);
    let timeoutId;
    const timeoutPromise = new Promise((_, reject) => {
        timeoutId = setTimeout(() => {
            reject(new Error(`timeout after ${timeoutMs}ms`));
        }, timeoutMs);
    });

    let acpOutcome;
    try {
        acpOutcome = await Promise.race([
            Promise.race([
                (async () => {
                    await send('initialize', {
                        protocolVersion: 1,
                        clientCapabilities: {
                            fs: { readTextFile: false, writeTextFile: false },
                            terminal: false,
                        },
                        clientInfo: { name: 'yawnbot-cursor-runner', version: '1.0.0' },
                    });

                    await send('authenticate', { methodId: 'cursor_login' });

                    const sessionParams = { cwd: opts.cwd, mcpServers: [] };
                    if (['agent', 'plan', 'ask'].includes(opts.mode)) {
                        sessionParams.mode = opts.mode;
                    }
                    const newSession = await send('session/new', sessionParams);
                    const sessionId = newSession.sessionId;

                    const pr = beginSend('session/prompt', {
                        sessionId,
                        prompt: [{ type: 'text', text: opts.prompt }],
                    });
                    promptTurnId = pr.id;
                    bumpIdle();
                    const promptResult = await pr.promise;

                    return {
                        stopReason: promptResult?.stopReason ?? null,
                        assistantText: collectedText.join(''),
                        rawPromptResult: promptResult,
                    };
                })(),
                spawnError,
            ]),
            timeoutPromise,
        ]);
    } finally {
        clearTimeout(timeoutId);
        clearTimeout(idleTimer);
        promptTurnId = null;
        try {
            child.stdin.end();
        } catch {
            /* ignore */
        }
        child.kill();
        await new Promise(r => setTimeout(r, 300));
    }

    return {
        stopReason: acpOutcome?.stopReason ?? null,
        assistantText: acpOutcome?.assistantText ?? collectedText.join(''),
        stderrTail: stderrChunks.join('').slice(-4000),
        acpRpcSummary: {
            askQuestionCount: askQuestionRpcCount,
            createPlanCount: createPlanRpcCount,
        },
    };
}

async function main() {
    const args = parseArgs(process.argv);
    const maxPrompt = parseInt(process.env.CURSOR_MAX_PROMPT_CHARS || '2000', 10);

    if (!args.cwd || !args.prompt) {
        emitJsonLine({
            ok: false,
            error: 'missing --cwd or --prompt',
        });
        return;
    }

    if (args.prompt.length > maxPrompt) {
        emitJsonLine({
            ok: false,
            error: `prompt exceeds CURSOR_MAX_PROMPT_CHARS (${maxPrompt})`,
        });
        return;
    }

    const resolvedCwd = path.resolve(args.cwd);
    if (!fs.existsSync(resolvedCwd) || !fs.statSync(resolvedCwd).isDirectory()) {
        emitJsonLine({ ok: false, error: `cwd is not a directory: ${resolvedCwd}` });
        return;
    }

    const allowed = process.env.CURSOR_LOCAL_REPO_DIR;
    if (!isWithinAllowedBase(resolvedCwd, allowed)) {
        emitJsonLine({
            ok: false,
            error: `cwd must be under CURSOR_LOCAL_REPO_DIR (${allowed})`,
        });
        return;
    }

    const timeoutMs =
        args.timeoutMs && !Number.isNaN(args.timeoutMs)
            ? args.timeoutMs
            : parseInt(process.env.CURSOR_TIMEOUT_MS || '600000', 10);

    const baseline = await createWorkingTreeBaseline(resolvedCwd);
    const baseRef = baseline.mode === 'baseline' && baseline.ref ? baseline.ref : null;

    let acpResult;
    try {
        acpResult = await runAcp({
            cwd: resolvedCwd,
            prompt: args.prompt,
            mode: args.mode || 'agent',
            timeoutMs,
        });
    } catch (e) {
        const git = await runGitSummarySince(resolvedCwd, baseRef);
        emitJsonLine({
            ok: false,
            error: e.message || String(e),
            git,
            snapshot: { baseline },
        });
        return;
    }

    const git = await runGitSummarySince(resolvedCwd, baseRef);
    emitJsonLine({
        ok: true,
        cwd: resolvedCwd,
        stopReason: acpResult.stopReason,
        assistantPreview: (acpResult.assistantText || '').slice(0, 4000),
        stderrTail: acpResult.stderrTail || '',
        acpRpcSummary: acpResult.acpRpcSummary || {
            askQuestionCount: 0,
            createPlanCount: 0,
        },
        git,
        snapshot: { baseline },
    });
}

main().catch(err => {
    emitJson({ ok: false, error: err.message || String(err) });
    process.exitCode = 1;
});
