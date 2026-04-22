import { existsSync } from 'node:fs';
import path from 'node:path';
import { MessageFlags, PermissionFlagsBits } from 'discord.js';
import type { ChatInputCommandInteraction, GuildMember } from 'discord.js';
import { packagedAudioDir } from '../../paths';
import { createAudioResourceFromHttpUrl, createAudioResourceFromLocalFile } from '../playable-audio-resource';
import { enqueueCustomTrack, withTimeout } from '../music-player';
import type { BotContext } from './bot-context';

const AUDIO_EXT = new Set(['.mp3', '.ogg', '.wav', '.m4a', '.opus', '.flac', '.webm']);
const SOUND_PREPARE_TIMEOUT_MS = 120_000;
const URL_MAX = 2000;

function resolvePackagedClip(raw: string): string | null {
  const name = path.basename(raw.trim());
  if (!name || name !== raw.trim()) {
    return null;
  }
  const ext = path.extname(name).toLowerCase();
  if (!AUDIO_EXT.has(ext)) {
    return null;
  }
  const root = path.resolve(packagedAudioDir());
  const full = path.resolve(root, name);
  const rootWithSep = root.endsWith(path.sep) ? root : root + path.sep;
  if (full !== root && !full.startsWith(rootWithSep)) {
    return null;
  }
  if (!existsSync(full)) {
    return null;
  }
  return full;
}

export async function handleSound(ctx: BotContext, interaction: ChatInputCommandInteraction): Promise<void> {
  if (!interaction.guild) {
    await interaction.reply({ content: '서버에서만 사용할 수 있습니다.', flags: MessageFlags.Ephemeral });
    return;
  }

  try {
    await interaction.deferReply();
  } catch (e) {
    const code = e && typeof e === 'object' && 'code' in e ? e.code : undefined;
    if (code === 10062) {
      console.warn('[sound] deferReply 10062 (Unknown interaction)');
      return;
    }
    throw e;
  }

  if (!interaction.member) {
    await interaction.editReply({ content: '멤버 정보를 불러올 수 없습니다. 잠시 후 다시 시도하세요.' });
    return;
  }

  const vc = (interaction.member as GuildMember).voice?.channel;
  if (!vc || !vc.isVoiceBased()) {
    await interaction.editReply({ content: '음성 채널에 들어간 뒤 `/music sound`를 사용하세요.' });
    return;
  }

  const botMember = interaction.guild?.members.me;
  if (!botMember) {
    await interaction.editReply({ content: '봇 멤버 정보를 불러올 수 없습니다.' });
    return;
  }

  const perms = vc.permissionsFor(botMember);
  if (!perms?.has([PermissionFlagsBits.Connect, PermissionFlagsBits.Speak, PermissionFlagsBits.ViewChannel])) {
    await interaction.editReply({
      content: '봇에게 해당 음성 채널 **보기·연결·말하기(Speak)** 권한이 필요합니다.',
    });
    return;
  }

  const attachment = interaction.options.getAttachment('file');
  const urlRaw = (interaction.options.getString('url') ?? '').trim();
  const clipRaw = (interaction.options.getString('clip') ?? '').trim();

  const sources = [attachment ? 'file' : null, urlRaw ? 'url' : null, clipRaw ? 'clip' : null].filter(Boolean);
  if (sources.length === 0) {
    await interaction.editReply({
      content:
        '`file`(첨부)·`url`·`clip` 중 **하나**만 지정하세요. 서버에 넣은 파일은 `resources/audio/` 에 두고 `/music sound clip:파일명.mp3` 로 재생할 수 있습니다.',
    });
    return;
  }
  if (sources.length > 1) {
    await interaction.editReply({ content: '`file` / `url` / `clip` 중 하나만 지정하세요.' });
    return;
  }

  let title: string;
  let load: () => Promise<import('@discordjs/voice').AudioResource>;

  if (attachment) {
    const ct = (attachment.contentType || '').toLowerCase();
    const name = attachment.name || '첨부';
    const ext = path.extname(name).toLowerCase();
    const looksAudio =
      ct.startsWith('audio/') ||
      AUDIO_EXT.has(ext) ||
      ct.includes('ogg') ||
      ct.includes('mpeg') ||
      ct.includes('wav');
    if (!looksAudio) {
      await interaction.editReply({
        content: '오디오로 보이는 첨부만 재생합니다. (mp3, wav, ogg 등 — `content-type`이 audio가 아니면 확장자를 확인합니다.)',
      });
      return;
    }
    title = `파일: ${name.slice(0, 80)}`;
    const href = attachment.url;
    load = () => createAudioResourceFromHttpUrl(href);
  } else if (urlRaw) {
    if (urlRaw.length > URL_MAX) {
      await interaction.editReply({ content: 'URL이 너무 깁니다.' });
      return;
    }
    let u;
    try {
      u = new URL(urlRaw);
    } catch {
      await interaction.editReply({ content: '올바른 http(s) URL이 아닙니다.' });
      return;
    }
    title = `URL: ${u.hostname}${u.pathname}`.slice(0, 120);
    load = () => createAudioResourceFromHttpUrl(urlRaw);
  } else {
    const resolved = resolvePackagedClip(clipRaw);
    if (!resolved) {
      await interaction.editReply({
        content: `클립을 찾을 수 없습니다. 봇 패키지 \`resources/audio/\` 에 ${[...AUDIO_EXT].join(', ')} 중 하나를 넣고, **파일명만** \`clip\`에 적어 주세요. (기본 샘플: \`demo.wav\`)`,
      });
      return;
    }
    title = `clip: ${path.basename(resolved)}`;
    load = () => createAudioResourceFromLocalFile(resolved);
  }

  await interaction.editReply({ content: '오디오 불러와서 대기열에 넣는 중…' });

  let result;
  try {
    result = await withTimeout(
      enqueueCustomTrack(vc, title, load, {
        notifyTextChannelId: interaction.channelId ?? undefined,
      }),
      SOUND_PREPARE_TIMEOUT_MS,
      '오디오 준비·재생',
    );
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    await interaction.editReply({ content: `실패: ${msg}` });
    return;
  }

  if (!result.ok) {
    await interaction.editReply({ content: `재생 준비 실패: ${result.error}` });
    return;
  }

  await interaction.editReply({
    content: result.started
      ? `**${title}** 재생을 시작합니다.`
      : `대기열 **${result.position}번**에 추가: **${title}**`,
  });
}
