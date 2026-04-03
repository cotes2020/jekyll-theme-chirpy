// @ts-nocheck
import { EmbedBuilder, MessageFlags } from 'discord.js';
import {
  formatCursorAcpRpcSummaryField,
  hasGitWorkingChanges,
  truncateDiscordDescription,
  truncateEmbedField,
  notifyDeferCompletion,
  startDeferElapsedTicker,
} from '@discord-bots/common';
import { discordAnswerCursorQuestion, getCursorMaxPromptChars, runCursorLocalRunner } from '../cursor-local';
import { resolveCursorRepoDirForSlash } from '../../paths';

export async function handleCursorEdit(ctx, interaction, userId) {
  const { gameData, isAdmin, cursorState } = ctx;
  if (!isAdmin(userId)) {
    await interaction.reply({ content: gameData.getMessage('Admin_AccessDenied_Desc'), flags: MessageFlags.Ephemeral });
    return;
  }
  const repoDir = resolveCursorRepoDirForSlash();
  if (!repoDir) {
    const triedEnv = process.env.CURSOR_LOCAL_REPO_DIR != null && String(process.env.CURSOR_LOCAL_REPO_DIR).trim();
    await interaction.reply({
      content: triedEnv
        ? '`CURSOR_LOCAL_REPO_DIR`이 가리키는 폴더가 없습니다. .env 경로를 고치거나, 비우면 이 레포 루트(자동)를 씁니다.'
        : 'Cursor 작업 폴더를 찾을 수 없습니다. yawnbot이 이 레포의 `apps/discord-bots/apps/yawnbot` 아래에 있어야 하거나, `.env`에 `CURSOR_LOCAL_REPO_DIR`을 직접 지정하세요.',
      flags: MessageFlags.Ephemeral,
    });
    return;
  }
  const promptText = interaction.options.getString('prompt');
  const modeOpt = interaction.options.getString('mode');
  const maxChars = getCursorMaxPromptChars();
  if (promptText.length > maxChars) {
    await interaction.reply({ content: `프롬프트가 너무 깁니다. 최대 ${maxChars}자까지입니다.`, flags: MessageFlags.Ephemeral });
    return;
  }
  if (cursorState.inFlight) {
    await interaction.reply({ content: '이미 Cursor 로컬 작업이 실행 중입니다. 잠시 후 다시 시도하세요.', flags: MessageFlags.Ephemeral });
    return;
  }
  await interaction.deferReply();
  const progressMin = Math.ceil(parseInt(process.env.CURSOR_TIMEOUT_MS || '600000', 10) / 60000);
  cursorState.inFlight = true;
  let stopDeferTicker: () => Promise<void> = async () => {};
  try {
    let liveAssistant = '';
    stopDeferTicker = await startDeferElapsedTicker(interaction, 'cursor', {
      progressMin,
      requestText: promptText,
      modeLabel: modeOpt || 'agent',
      liveAssistantText: () => liveAssistant,
    });
    const { json, code, err } = await runCursorLocalRunner(
      repoDir,
      promptText,
      modeOpt || 'agent',
      (chunk) => {
        liveAssistant += chunk;
        const maxLive = parseInt(process.env.CURSOR_LIVE_PREVIEW_CHARS || '700', 10);
        if (liveAssistant.length > maxLive) liveAssistant = liveAssistant.slice(-maxLive);
      },
      (q) => discordAnswerCursorQuestion(interaction, q),
    );
    await stopDeferTicker();
    stopDeferTicker = async () => {};
    if (!json.ok) {
      const git = json.git || {};
      const embed = new EmbedBuilder()
        .setTitle('Cursor 로컬 실행 실패')
        .setDescription(
          truncateDiscordDescription(
            `**📝 요청**\n\`\`\`\n${truncateEmbedField(promptText, 900)}\n\`\`\`\n\n` +
              `**오류**\n${String(json.error || 'unknown').slice(0, 2800)}`,
          ),
        )
        .setColor(0xf44336)
        .addFields(
          { name: 'exit', value: String(code), inline: true },
          { name: 'git repo', value: git.isRepo ? 'yes' : 'no', inline: true },
          { name: 'mode', value: `\`${modeOpt || 'agent'}\``, inline: true },
        );
      if (hasGitWorkingChanges(git)) {
        if (String(git.statusPorcelain || '').trim()) {
          embed.addFields({
            name: '변경 파일 (status)',
            value: `\`\`\`\n${String(git.statusPorcelain).slice(0, 900)}\n\`\`\``,
          });
        }
        if (String(git.diffStat || '').trim()) {
          embed.addFields({ name: 'diff --stat', value: `\`\`\`\n${String(git.diffStat).slice(0, 900)}\n\`\`\`` });
        }
        if (String(git.diffPreview || '').trim()) {
          embed.addFields({ name: 'diffPreview', value: `\`\`\`\n${String(git.diffPreview).slice(0, 900)}\n\`\`\`` });
        }
      }
      if (err) embed.setFooter({ text: err.slice(0, 500) });
      await interaction.editReply({ content: null, embeds: [embed] });
      await notifyDeferCompletion(interaction, { ok: false, kind: 'cursor' });
      return;
    }
    const g = json.git || {};
    const descParts = [];
    if (json.stopReason != null) descParts.push(`**stopReason:** ${json.stopReason}`);
    if (json.assistantPreview) descParts.push(json.assistantPreview.slice(0, 2600));
    const embed = new EmbedBuilder()
      .setTitle('Cursor 로컬 실행 완료')
      .setDescription(
        truncateDiscordDescription(
          `**📝 요청**\n\`\`\`\n${truncateEmbedField(promptText, 900)}\n\`\`\`\n\n` +
            `**에이전트 응답**\n${descParts.join('\n\n') || '(응답 본문 없음)'}`,
        ),
      )
      .setColor(0x4caf50)
      .addFields(
        { name: '모드', value: `\`${modeOpt || 'agent'}\``, inline: true },
        { name: '작업 경로', value: `\`${String(json.cwd || repoDir).slice(0, 900)}\`` },
        {
          name: '질문·플랜 (ACP)',
          value: formatCursorAcpRpcSummaryField(json.acpRpcSummary),
          inline: false,
        },
      );
    if (hasGitWorkingChanges(g)) {
      if (String(g.statusPorcelain || '').trim()) {
        embed.addFields({
          name: '변경 파일 (status)',
          value: `\`\`\`\n${String(g.statusPorcelain).slice(0, 900)}\n\`\`\``,
        });
      }
      if (String(g.diffStat || '').trim()) {
        embed.addFields({ name: 'diff --stat', value: `\`\`\`\n${String(g.diffStat).slice(0, 900)}\n\`\`\`` });
      }
      if (String(g.diffPreview || '').trim()) {
        embed.addFields({ name: 'diffPreview', value: `\`\`\`\n${String(g.diffPreview).slice(0, 900)}\n\`\`\`` });
      }
    }
    if (json.stderrTail) embed.addFields({ name: 'agent stderr (tail)', value: `\`\`\`\n${String(json.stderrTail).slice(0, 800)}\n\`\`\`` });
    await interaction.editReply({ content: null, embeds: [embed] });
    await notifyDeferCompletion(interaction, { ok: true, kind: 'cursor' });
  } catch (e) {
    await stopDeferTicker();
    stopDeferTicker = async () => {};
    await interaction.editReply({
      content: null,
      embeds: [
        new EmbedBuilder()
          .setTitle('Cursor 로컬 실행 예외')
          .setDescription(
            truncateDiscordDescription(
              `**📝 요청**\n\`\`\`\n${truncateEmbedField(promptText, 900)}\n\`\`\`\n\n` + `**오류**\n${String(e.message).slice(0, 1800)}`,
            ),
          )
          .setColor(0xf44336),
      ],
    });
    await notifyDeferCompletion(interaction, { ok: false, kind: 'cursor' });
  } finally {
    await stopDeferTicker();
    cursorState.inFlight = false;
  }
}

export async function handleYawn(ctx, interaction) {
  const { geminiModel } = ctx;
  if (!geminiModel) {
    await interaction.reply({ content: 'Gemini API가 설정되지 않았습니다.', flags: MessageFlags.Ephemeral });
    return;
  }
  const prompt = interaction.options.getString('질문');
  await interaction.deferReply();
  let stopGeminiTicker: () => Promise<void> = async () => {};
  try {
    stopGeminiTicker = await startDeferElapsedTicker(interaction, 'gemini', { requestText: prompt });
    const result = await geminiModel.generateContent({
      contents: [
        {
          role: 'user',
          parts: [
            {
              text: `시스템: 너는 'YawnBot'이라는 이름의 활기차고 재치 있는 디스코드 봇이야. 사용자의 질문에 친절하고 유머러스하게 대답해줘.\n\n사용자: ${prompt}`,
            },
          ],
        },
      ],
    });
    await stopGeminiTicker();
    stopGeminiTicker = async () => {};
    const response = result.response.text();
    const embed = new EmbedBuilder()
      .setTitle('YawnBot AI Response')
      .setDescription(
        truncateDiscordDescription(
          `**📝 질문**\n\`\`\`\n${truncateEmbedField(prompt, 800)}\n\`\`\`\n\n` + `**💬 답변**\n${response.slice(0, 3000)}`,
        ),
      )
      .setColor(0x4285f4)
      .setFooter({ text: 'Powered by Google Gemini' })
      .setTimestamp();
    await interaction.editReply({ content: null, embeds: [embed] });
    await notifyDeferCompletion(interaction, { ok: true, kind: 'gemini' });
  } catch (e) {
    await stopGeminiTicker();
    stopGeminiTicker = async () => {};
    await interaction.editReply({
      content: null,
      embeds: [
        new EmbedBuilder()
          .setTitle('Gemini 오류')
          .setDescription(
            truncateDiscordDescription(
              `**📝 질문**\n\`\`\`\n${truncateEmbedField(prompt, 800)}\n\`\`\`\n\n` + `**오류**\n${String(e.message).slice(0, 1800)}`,
            ),
          )
          .setColor(0xf44336),
      ],
    });
    await notifyDeferCompletion(interaction, { ok: false, kind: 'gemini' });
  } finally {
    await stopGeminiTicker();
  }
}

