import path from 'path';
import http from 'node:http';
import { config } from 'dotenv';
import {
  AttachmentBuilder,
  Client,
  EmbedBuilder,
  GatewayIntentBits,
  type SendableChannels,
} from 'discord.js';
import { fetchHnTopStories, type HnStoryLine } from './geeknews/fetchHnTop';
import type { PublisherSaleAssetInfo } from './unity/publisherSaleAsset';
import { fetchPublisherSaleAssetInfo } from './unity/publisherSaleAsset';

// cwd와 무관하게 이 패키지 루트의 .env만 사용 (yawnbot/.env와 분리)
config({ path: path.join(__dirname, '..', '.env') });

const TOKEN = process.env.ATKUP_DISCORD_TOKEN;
const TARGET_CHANNEL_ID = process.env.ATKUP_TARGET_CHANNEL_ID;
const CHECK_INTERVAL_MIN = parseInt(process.env.ATKUP_CHECK_INTERVAL_MIN || '60', 10);

const HN_COLOR = 0xff6600;

/** 마크다운 링크용: 제목에 [ ] 가 있으면 깨지므로 제거 */
function sanitizeTitleForMdLink(title: string): string {
  return title.replace(/[\[\]]/g, '').trim() || '제목';
}

function buildGeekNewsEmbed(lines: HnStoryLine[]): EmbedBuilder {
  const desc = lines
    .map((s, i) => {
      const t = sanitizeTitleForMdLink(s.title);
      return `${i + 1}. [${t}](${s.href}) · ${s.score}pt · ${s.host} · ${s.by}`;
    })
    .join('\n');

  return new EmbedBuilder()
    .setTitle('📰 Hacker News · 긱 뉴스')
    .setDescription(desc.slice(0, 4090))
    .setColor(HN_COLOR)
    .setFooter({ text: 'ATKUp · GeekNews · Lobsters' });
}

async function sendGeekNewsToChannel(channel: SendableChannels, count: number): Promise<number> {
  const stories = await fetchHnTopStories(count);
  if (stories.length === 0) {
    await channel.send({ content: 'Hacker News에서 가져온 글이 없습니다.' });
    return 0;
  }
  const embed = buildGeekNewsEmbed(stories);
  await channel.send({ embeds: [embed] });
  return stories.length;
}

const EMBED_IMAGE_ATTACHMENT = 'atkup-publisher-og.jpg';

/** Discord가 외부 CDN 이미지 프록시에 실패하는 경우가 있어, 가능하면 첨부로 넣는다. */
async function tryOgImageAttachment(imageUrl: string | null): Promise<AttachmentBuilder | null> {
  if (!imageUrl) return null;
  try {
    const res = await fetch(imageUrl, {
      headers: {
        'User-Agent':
          'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36',
      },
    });
    if (!res.ok) return null;
    const buf = Buffer.from(await res.arrayBuffer());
    if (buf.length < 64 || buf.length > 8 * 1024 * 1024) return null;
    return new AttachmentBuilder(buf, { name: EMBED_IMAGE_ATTACHMENT });
  } catch {
    return null;
  }
}

function buildPublisherSaleEmbed(info: PublisherSaleAssetInfo, imageFromAttachment: boolean) {
  const title = info.assetName ? `🎁 이번 주 Unity 무료 에셋: ${info.assetName}` : '🎁 이번 주 Unity 무료 에셋';

  const lines: string[] = [];
  if (info.couponCode) lines.push(`✅ **쿠폰 코드**: \`${info.couponCode}\``);
  if (info.promoText) lines.push(`ℹ️ ${info.promoText}`);

  const embed = new EmbedBuilder()
    .setTitle(title)
    .setDescription(lines.join('\n') || ' ')
    .setColor(0x00bcd4)
    .setFooter({ text: 'ATKUp' });
  if (info.assetUrl) embed.setURL(info.assetUrl);
  if (imageFromAttachment) {
    embed.setImage(`attachment://${EMBED_IMAGE_ATTACHMENT}`);
  } else if (info.imageUrl) {
    embed.setImage(info.imageUrl);
  }
  return embed;
}

async function sendPublisherSaleToChannel(channel: SendableChannels, info: PublisherSaleAssetInfo) {
  const ogAttach = await tryOgImageAttachment(info.imageUrl);
  const embed = buildPublisherSaleEmbed(info, !!ogAttach);
  const files = ogAttach ? [ogAttach] : [];
  const content =
    info.assetUrl != null && info.assetUrl.length > 0
      ? `에셋 페이지(탭에서 열기): ${info.assetUrl}`
      : undefined;
  try {
    await channel.send({ content, embeds: [embed], files });
  } catch (err: unknown) {
    if (ogAttach && files.length > 0) {
      console.warn('[ATKUp] 이미지 첨부 전송 실패(권한·용량 등), 외부 URL 이미지로 재시도:', err);
      const fallback = buildPublisherSaleEmbed(info, false);
      await channel.send({ content, embeds: [fallback] });
      return;
    }
    throw err;
  }
}

/** KarmoLab 서버 모니터 등에서 프로세스 생존 확인용 (127.0.0.1만). `ATKUP_HEALTH_PORT=0` 이면 끔 */
function startHealthServer(): void {
  const raw = (process.env.ATKUP_HEALTH_PORT ?? '').trim().toLowerCase();
  if (raw === '0' || raw === 'off' || raw === 'false') return;

  const port = parseInt(process.env.ATKUP_HEALTH_PORT || '4616', 10);
  if (!Number.isFinite(port) || port < 1 || port > 65535) {
    console.warn('[ATKUp] ATKUP_HEALTH_PORT가 올바르지 않아 헬스 HTTP를 켜지 않습니다.');
    return;
  }

  const server = http.createServer((req, res) => {
    const u = req.url?.split('?')[0] ?? '';
    if (req.method === 'GET' && u === '/health') {
      res.writeHead(200, { 'Content-Type': 'text/plain; charset=utf-8' });
      res.end('ok');
      return;
    }
    res.writeHead(404);
    res.end();
  });

  server.on('error', (err: NodeJS.ErrnoException) => {
    if (err.code === 'EADDRINUSE') {
      console.error(`[ATKUp] 헬스 포트 ${port} 사용 중 — ATKUP_HEALTH_PORT를 바꾸거나 충돌 프로세스를 종료하세요.`);
    } else {
      console.error('[ATKUp] 헬스 HTTP 서버 오류:', err.message);
    }
  });

  server.listen(port, '127.0.0.1', () => {
    console.log(`[ATKUp] 헬스: http://127.0.0.1:${port}/health`);
  });
}

function startAtkupBot() {
  startHealthServer();

  const client = new Client({
    intents: [GatewayIntentBits.Guilds],
  });

  let lastSentCoupon: string | null = null;

  async function pollPublisherSaleOnce({ force = false }: { force?: boolean } = {}) {
    console.log('[ATKUp] Unity 무료 에셋 정보 확인 중...');

    let info: PublisherSaleAssetInfo | null;
    try {
      info = await fetchPublisherSaleAssetInfo();
    } catch (err: any) {
      console.error('[ATKUp] Unity 페이지 요청/파싱 실패:', err?.message ?? err);
      return;
    }

    if (!info) {
      console.log('[ATKUp] 현재 활성화된 무료 에셋 증정이 없는 것 같습니다.');
      return;
    }

    if (!force && info.couponCode && lastSentCoupon === info.couponCode) {
      console.log('[ATKUp] 이미 전송했던 쿠폰 코드입니다. 건너뜀:', info.couponCode);
      return;
    }

    const channel = await client.channels.fetch(TARGET_CHANNEL_ID!).catch(() => null);
    if (!channel?.isSendable()) {
      console.error('[ATKUp] 채널을 찾을 수 없거나 메시지를 보낼 수 없습니다:', TARGET_CHANNEL_ID);
      return;
    }

    await sendPublisherSaleToChannel(channel, info);
    console.log('[ATKUp] 무료 에셋 알림 전송 완료.');

    if (info.couponCode) {
      lastSentCoupon = info.couponCode;
    }
  }

  client.once('clientReady', async () => {
    console.log(`\n  🎁  ATKUp`);
    console.log('  ─────────────────────────');
    console.log(`  로그인: ${client.user?.tag}`);
    console.log('');

    await pollPublisherSaleOnce();

    const intervalMs = Math.max(5, CHECK_INTERVAL_MIN) * 60 * 1000;
    setInterval(() => {
      void pollPublisherSaleOnce();
    }, intervalMs);
  });

  client.on('interactionCreate', async (interaction) => {
    if (!interaction.isChatInputCommand()) return;
    if (interaction.commandName !== 'atkup') return;

    const sub = interaction.options.getSubcommand(true);

    if (sub === 'news') {
      await interaction.deferReply({ ephemeral: true });
      const count = interaction.options.getInteger('count') ?? 10;

      const channel = await client.channels.fetch(TARGET_CHANNEL_ID!).catch(() => null);
      if (!channel?.isSendable()) {
        await interaction.editReply(`채널을 찾을 수 없거나 메시지를 보낼 수 없습니다: ${TARGET_CHANNEL_ID}`);
        return;
      }

      let n: number;
      try {
        n = await sendGeekNewsToChannel(channel, count);
      } catch (err: any) {
        await interaction.editReply(`긱 뉴스 가져오기 실패: ${err?.message ?? err}`);
        return;
      }

      await interaction.editReply(`ATKUp · Hacker News 글 ${n}개를 알림 채널에 보냈습니다.`);
      return;
    }

    if (sub === 'unity') {
      const force = interaction.options.getBoolean('force') || false;
      await interaction.deferReply({ ephemeral: true });

      let info: PublisherSaleAssetInfo | null;
      try {
        info = await fetchPublisherSaleAssetInfo();
      } catch (err: any) {
        await interaction.editReply(`가져오기 실패: ${err?.message ?? err}`);
        return;
      }

      if (!info) {
        await interaction.editReply('현재 활성화된 무료 에셋 증정이 없는 것 같습니다.');
        return;
      }

      const coupon = info.couponCode || '(쿠폰 없음)';
      if (!force && info.couponCode && lastSentCoupon === info.couponCode) {
        await interaction.editReply(
          `이미 전송했던 쿠폰 코드라서 건너뜁니다: \`${coupon}\` (/atkup unity 명령에서 force 옵션을 켜면 강제 전송)`,
        );
        return;
      }

      const channel = await client.channels.fetch(TARGET_CHANNEL_ID!).catch(() => null);
      if (!channel?.isSendable()) {
        await interaction.editReply(`채널을 찾을 수 없거나 메시지를 보낼 수 없습니다: ${TARGET_CHANNEL_ID}`);
        return;
      }

      await sendPublisherSaleToChannel(channel, info);
      if (info.couponCode) lastSentCoupon = info.couponCode;

      await interaction.editReply(`ATKUp · 전송 완료: \`${coupon}\``);
    }
  });

  client.login(TOKEN!).catch((err) => {
    console.error('[ATKUp] 로그인 실패:', err);
    process.exit(1);
  });
}

if (!TOKEN) {
  console.warn(
    '[ATKUp] ATKUP_DISCORD_TOKEN 없음 — ATKUp 비활성. apps/atkup-bot/.env 를 설정하세요. (YawnBot과 별도 토큰)',
  );
  console.warn('[ATKUp] `npm run dev` 만 쓸 때는 이 경고는 무시해도 됩니다. YawnBot만 개발하려면: npm run dev:yawnbot');
  setInterval(() => {}, 60 * 60 * 1000);
} else if (!TARGET_CHANNEL_ID) {
  console.error('[ATKUp] 알림을 보낼 채널 ID가 없습니다. .env의 ATKUP_TARGET_CHANNEL_ID를 설정하세요.');
  process.exit(1);
} else {
  startAtkupBot();
}
