/**
 * reactions.ts — messageReactionAdd 이벤트 핸들러
 *
 * 지원 이모지:
 *   🔄  봇 이미지 메시지 → 같은 프롬프트로 재생성
 *   ⭐  봇 응답 → 캐릭터 메모리 하이라이트 저장
 *   ⏭️  음악 스킵
 *   🔀  음악 셔플
 *   ⏸️  음악 일시정지/재개
 *   ❓  해당 유저의 과거 메시지 랜덤 인용
 *   기타  봇 응답에 달린 이모지 → 캐릭터가 짧게 반응
 */
import fs from 'fs';
import path from 'path';
import type { MessageReaction, PartialMessageReaction, User, PartialUser, TextChannel, DMChannel } from 'discord.js';
import { AttachmentBuilder } from 'discord.js';
import { generateAssistantText, generateImageFromEnvWithOptions } from 'karmolab-ai/node';
import { CharacterService } from '../services/character-service';
import type { BotContext } from './slash/bot-context';
import { skipTrack, shuffleWaitingQueue, pauseToggleMusic } from './music-player';

// 전용 기능 이모지 (이것 외 봇 메시지 이모지 → 캐릭터 반응)
const COMMAND_EMOJIS = new Set(['🔄', '⭐', '⏭️', '🔀', '⏸️', '❓']);

// ⭐ 중복 방지: 최근 처리한 메시지 ID 캐시
const highlightedMessageIds = new Set<string>();

export async function handleReaction(
  ctx: BotContext,
  reaction: MessageReaction | PartialMessageReaction,
  user: User | PartialUser,
): Promise<void> {
  if (user.bot) return;

  // partial 해소
  try {
    if (reaction.partial) await reaction.fetch();
    if (reaction.message.partial) await reaction.message.fetch();
  } catch {
    return;
  }

  const message = reaction.message;
  const emoji = reaction.emoji.name ?? '';
  const isFromBot = message.author?.id === ctx.client.user?.id;
  const guildId = message.guildId;

  // ── 음악 제어 (길드 한정) ─────────────────────────────────────────
  if (guildId) {
    if (emoji === '⏭️') {
      const skipped = skipTrack(guildId);
      console.log(`[Reaction] ⏭️ 스킵 ${skipped ? '성공' : '실패'} (guild=${guildId})`);
      return;
    }
    if (emoji === '🔀') {
      const count = shuffleWaitingQueue(guildId);
      console.log(`[Reaction] 🔀 셔플 ${count}곡 (guild=${guildId})`);
      return;
    }
    if (emoji === '⏸️') {
      const result = pauseToggleMusic(guildId);
      console.log(`[Reaction] ⏸️ ${result || '재생 중 아님'} (guild=${guildId})`);
      return;
    }
  }

  // ── 이하 오너 전용 ────────────────────────────────────────────────
  if (!ctx.isOwner(user.id)) return;

  // ── ❓ 랜덤 인용구 ────────────────────────────────────────────────
  if (emoji === '❓') {
    await handleRandomQuote(ctx, reaction, user);
    return;
  }

  // ── 봇 메시지 전용 기능 ───────────────────────────────────────────
  if (!isFromBot) return;

  if (emoji === '🔄') {
    await handleImageRegenerate(ctx, reaction);
    return;
  }

  if (emoji === '⭐') {
    await handleMemoryHighlight(ctx, reaction, user);
    return;
  }

  // ── 캐릭터 이모지 반응 (COMMAND_EMOJIS 제외한 나머지) ──────────────
  if (!COMMAND_EMOJIS.has(emoji)) {
    await handleCharacterReaction(ctx, reaction, user, emoji);
  }
}

// ── 🔄 이미지 재생성 ────────────────────────────────────────────────

async function handleImageRegenerate(
  ctx: BotContext,
  reaction: MessageReaction | PartialMessageReaction,
): Promise<void> {
  const message = reaction.message;
  if (!message.attachments.size && !message.embeds.length) return;

  // embed description을 프롬프트로 사용 (runImageGeneration이 거기 넣음)
  const embed = message.embeds[0];
  const prompt = embed?.description?.trim();
  if (!prompt) {
    console.log('[Reaction:🔄] 프롬프트 없음 (embed description 비어있음)');
    return;
  }

  console.log(`[Reaction:🔄] 재생성 시작: "${prompt.slice(0, 60)}..."`);
  try {
    const { images, modelId } = await generateImageFromEnvWithOptions(process.env, prompt, { sampleCount: 1 });
    if (!images.length) return;
    const img = images[0];
    const ext = (img.mimeType.split('/')[1] || 'png').replace(/[^a-z0-9]/gi, '');
    const attachment = new AttachmentBuilder(img.buffer, { name: `regen.${ext}` });
    const ch = message.channel as TextChannel | DMChannel;
    await ch.send({ content: `🔄 재생성 (${modelId})`, files: [attachment], reply: { messageReference: message.id } });
    console.log(`[Reaction:🔄] 완료 (모델=${modelId})`);
  } catch (e) {
    console.error('[Reaction:🔄] 실패:', e instanceof Error ? e.message : e);
  }
}

// ── ⭐ 메모리 하이라이트 ────────────────────────────────────────────

async function handleMemoryHighlight(
  ctx: BotContext,
  reaction: MessageReaction | PartialMessageReaction,
  user: User | PartialUser,
): Promise<void> {
  const message = reaction.message;
  if (highlightedMessageIds.has(message.id)) return;

  const content = message.content?.trim();
  if (!content || !ctx.characterService || !ctx.getMemory) return;

  const isDM = message.channel.isDMBased();
  const channelKey = CharacterService.channelKey({
    isDM,
    userId: user.id!,
    channelId: message.channel.id,
  });
  const card = ctx.characterService.resolveCard(channelKey);
  if (!card) return;

  const memory = ctx.getMemory(card.slug);
  const fact = `[하이라이트] ${content.slice(0, 200)}`;
  memory.appendHotMemory(fact);
  highlightedMessageIds.add(message.id);
  console.log(`[Reaction:⭐] 메모리 저장 (slug=${card.slug}): ${fact.slice(0, 60)}`);
  await reaction.message.react('✅').catch(() => {});
}

// ── ❓ 랜덤 인용구 ─────────────────────────────────────────────────

async function handleRandomQuote(
  ctx: BotContext,
  reaction: MessageReaction | PartialMessageReaction,
  user: User | PartialUser,
): Promise<void> {
  if (!ctx.characterService || !ctx.getMemory) return;

  const isDM = reaction.message.channel.isDMBased();
  const channelKey = CharacterService.channelKey({
    isDM,
    userId: user.id!,
    channelId: reaction.message.channel.id,
  });
  const card = ctx.characterService.resolveCard(channelKey);
  if (!card) return;

  const memory = ctx.getMemory(card.slug);
  const logsDir = path.join(memory['logsDir'] as string);

  try {
    const files = fs.existsSync(logsDir)
      ? fs.readdirSync(logsDir).filter((f) => f.endsWith('.md'))
      : [];
    if (!files.length) return;

    // 모든 로그에서 유저 발화 줄 수집
    const userLines: string[] = [];
    for (const file of files) {
      const raw = fs.readFileSync(path.join(logsDir, file), 'utf-8');
      for (const line of raw.split('\n')) {
        // 형식: [HH:MM DM] **나**: 내용
        const match = line.match(/\*\*나\*\*:\s*(.+)/);
        if (match?.[1]?.trim()) userLines.push(match[1].trim());
      }
    }

    if (!userLines.length) return;

    const quote = userLines[Math.floor(Math.random() * userLines.length)];
    const ch = reaction.message.channel as TextChannel | DMChannel;
    await ch.send(`> "${quote}"`);
    console.log(`[Reaction:❓] 인용: "${quote.slice(0, 60)}"`);
  } catch (e) {
    console.error('[Reaction:❓] 실패:', e instanceof Error ? e.message : e);
  }
}

// ── 🎲 캐릭터 이모지 반응 ──────────────────────────────────────────

async function handleCharacterReaction(
  ctx: BotContext,
  reaction: MessageReaction | PartialMessageReaction,
  user: User | PartialUser,
  emoji: string,
): Promise<void> {
  if (!ctx.characterService) return;

  const message = reaction.message;
  const isDM = message.channel.isDMBased();
  const channelKey = CharacterService.channelKey({
    isDM,
    userId: user.id!,
    channelId: message.channel.id,
  });
  const card = ctx.characterService.resolveCard(channelKey);
  if (!card) return;

  const originalContent = message.content?.slice(0, 300) || '(내용 없음)';
  const prompt =
    `유저가 네 메시지에 ${emoji} 이모지로 반응했어.\n` +
    `네 메시지: "${originalContent}"\n` +
    `이 이모지에 어울리는 짧은 반응을 1문장으로, 한국어로 자연스럽게 보내줘.`;

  try {
    const { text } = await generateAssistantText(process.env, prompt, { systemInstruction: card.body });
    const ch = message.channel as TextChannel | DMChannel;
    await ch.send({ content: text.trim().slice(0, 500), reply: { messageReference: message.id } });
    console.log(`[Reaction:🎲] ${emoji} → 캐릭터 반응 전송 (slug=${card.slug})`);
  } catch (e) {
    console.error('[Reaction:🎲] 실패:', e instanceof Error ? e.message : e);
  }
}
