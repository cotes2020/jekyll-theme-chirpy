// @ts-nocheck
import { spawn, execFile } from 'child_process';
import type { ChatInputCommandInteraction, StringSelectMenuInteraction } from 'discord.js';
import { ActionRowBuilder, EmbedBuilder, MessageFlags, StringSelectMenuBuilder } from 'discord.js';
import { cursorRunnerScript, resolveCursorRepoDir } from '../paths';
import { truncateDiscordDescription } from '@discord-bots/common';

const CURSOR_RUNNER_PATH = cursorRunnerScript();

export function getCursorMaxPromptChars() {
  return parseInt(process.env.CURSOR_MAX_PROMPT_CHARS || '2000', 10);
}

export async function discordAnswerCursorQuestion(
  interaction: ChatInputCommandInteraction,
  payload: { rpcId?: unknown; params?: Record<string, unknown> },
) {
  const params = payload.params || {};
  const raw = (params.options ?? params.choices ?? []) as unknown[];
  const lines = Array.isArray(raw) ? raw : [];
  const selectOptions = lines.slice(0, 25).map((o, i) => {
    if (typeof o === 'string') {
      return { label: o.slice(0, 100), value: String(i) };
    }
    const label = String((o as any).label ?? (o as any).title ?? (o as any).text ?? `선택 ${i + 1}`).slice(0, 100);
    const out: any = { label, value: String(i) };
    if ((o as any).description != null) out.description = String((o as any).description).slice(0, 100);
    return out;
  });
  if (selectOptions.length === 0) {
    return { cancelled: true };
  }
  const heading = (params as any).title || (params as any).question || '에이전트 질문';
  const embed = new EmbedBuilder().setTitle('에이전트 질문').setDescription(truncateDiscordDescription(String(heading))).setColor(0x5865f2);
  const customId = `cursor_q_${String(payload.rpcId)}`;
  const menu = new StringSelectMenuBuilder().setCustomId(customId).setPlaceholder('답을 선택하세요').addOptions(selectOptions as any);
  const row = new ActionRowBuilder().addComponents(menu as any);
  const reply: any = await (interaction as any).followUp({
    embeds: [embed],
    components: [row],
    flags: MessageFlags.Ephemeral,
    fetchReply: true,
  });
  const uid = interaction.user.id;
  try {
    const comp = (await reply.awaitMessageComponent({
      filter: (i: any): i is StringSelectMenuInteraction => i.user.id === uid && i.customId === customId,
      time: 600_000,
    })) as StringSelectMenuInteraction;
    const idx = parseInt((comp as any).values[0], 10);
    await (comp as any).update({ components: [] });
    return { selectedIndex: idx };
  } catch {
    try {
      await reply.edit({ components: [] });
    } catch {
      /* ignore */
    }
    return { cancelled: true };
  }
}

export function runCursorLocalRunner(cwd, prompt, mode, onProgress, onQuestion) {
  const innerTimeoutMs = parseInt(process.env.CURSOR_TIMEOUT_MS || '600000', 10);
  const outerGraceMs = parseInt(process.env.CURSOR_RUNNER_GRACE_MS || '120000', 10);
  const hardCapMs = Math.max(60000, innerTimeoutMs + outerGraceMs);
  const resolvedCwd = resolveCursorRepoDir(cwd);
  const resolvedAllow = process.env.CURSOR_LOCAL_REPO_DIR
    ? resolveCursorRepoDir(process.env.CURSOR_LOCAL_REPO_DIR)
    : resolvedCwd;

  return new Promise((resolve, reject) => {
    const args = [
      CURSOR_RUNNER_PATH,
      '--cwd',
      resolvedCwd,
      '--prompt',
      prompt,
      '--mode',
      mode || 'agent',
      '--timeoutMs',
      String(innerTimeoutMs),
    ];
    const child = spawn(process.execPath, args, {
      env: {
        ...process.env,
        CURSOR_LOCAL_REPO_DIR: resolvedAllow,
        CURSOR_GIT_SNAPSHOT: process.env.CURSOR_GIT_SNAPSHOT || 'baseline',
      },
      windowsHide: true,
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    let out = '';
    let err = '';
    const maxOut = 12 * 1024 * 1024;
    let settled = false;
    let resultJson: any = null;
    let outBuf = '';

    const killTree = () => {
      if (!child.pid) return;
      try {
        if (process.platform === 'win32') {
          execFile('taskkill', ['/PID', String(child.pid), '/T', '/F'], { windowsHide: true }, () => {});
        } else {
          child.kill('SIGKILL');
        }
      } catch {
        try {
          child.kill('SIGKILL');
        } catch {
          /* ignore */
        }
      }
    };

    const hardTimer = setTimeout(() => {
      if (settled) return;
      settled = true;
      killTree();
      reject(new Error(`러너 시간 초과 (${Math.round(hardCapMs / 1000)}초). Cursor 에이전트가 멈췄거나 종료되지 않았습니다.`));
    }, hardCapMs);

    child.stdout.on('data', (chunk) => {
      const s = chunk.toString();
      if (out.length < maxOut) out += s;
      outBuf += s;
      while (true) {
        const idx = outBuf.indexOf('\n');
        if (idx === -1) break;
        const line = outBuf.slice(0, idx).trim();
        outBuf = outBuf.slice(idx + 1);
        if (!line) continue;
        try {
          const msg = JSON.parse(line);
          if (msg && msg.type === 'assistant_chunk' && typeof msg.text === 'string') {
            if (typeof onProgress === 'function') onProgress(msg.text);
          } else if (msg && msg.type === 'cursor_question' && typeof onQuestion === 'function') {
            void (async () => {
              try {
                const ans = await onQuestion(msg);
                const rid = msg.rpcId;
                const payloadOut =
                  ans && ans.cancelled === true ? { rpcId: rid, cancelled: true } : { rpcId: rid, selectedIndex: ans.selectedIndex };
                if (child.stdin && !child.stdin.destroyed) {
                  child.stdin.write(`${JSON.stringify(payloadOut)}\n`);
                }
              } catch (e: any) {
                console.error('[cursor question]', e?.message ?? e);
                if (child.stdin && !child.stdin.destroyed) {
                  child.stdin.write(`${JSON.stringify({ rpcId: msg.rpcId, cancelled: true })}\n`);
                }
              }
            })();
          } else if (msg && typeof msg.ok === 'boolean') {
            resultJson = msg;
          }
        } catch {
          /* ignore non-JSON */
        }
      }
    });
    child.stderr.on('data', (chunk) => {
      if (err.length < 256 * 1024) err += chunk.toString();
    });
    child.on('error', (e) => {
      if (settled) return;
      settled = true;
      clearTimeout(hardTimer);
      reject(e);
    });
    child.on('close', (code) => {
      if (settled) return;
      settled = true;
      clearTimeout(hardTimer);
      try {
        const tail = outBuf.trim();
        if (tail) {
          try {
            const msg = JSON.parse(tail);
            if (msg && typeof msg.ok === 'boolean') resultJson = msg;
          } catch {
            /* ignore */
          }
        }
        if (resultJson && typeof resultJson.ok === 'boolean') {
          resolve({ json: resultJson, code, err });
          return;
        }
        const trimmed = out.trim();
        const lastLine = trimmed.includes('\n') ? trimmed.split('\n').pop()!.trim() : trimmed;
        const json = JSON.parse(lastLine);
        resolve({ json, code, err });
      } catch (e: any) {
        reject(new Error(`runner JSON parse: ${e?.message ?? e}; exit=${code}; stderr=${String(err).slice(0, 2000)}`));
      }
    });
  });
}

