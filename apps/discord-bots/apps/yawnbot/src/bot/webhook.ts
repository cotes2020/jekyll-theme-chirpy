// @ts-nocheck
import express from 'express';
import { EmbedBuilder } from 'discord.js';
import type { Client } from 'discord.js';
import { parseCommaSeparatedEnv } from '@discord-bots/common';
import type { GameDataService } from '../services/gamedata';

export function createGithubWebhookApp(client: Client, gameData: GameDataService) {
  const app = express();
  app.use(express.json());

  app.post('/webhook/github', async (req, res) => {
    try {
      const event = req.headers['x-github-event'];
      const payload = req.body;
      console.log(`[Webhook] Received: ${event}`);

      const channelIds = parseCommaSeparatedEnv(process.env.GITHUB_WEBHOOK_CHANNEL_ID);
      if (channelIds.length === 0) {
        console.warn('[Webhook] GITHUB_WEBHOOK_CHANNEL_ID 없음 — 디스코드 전송 생략 (200 응답)');
        res.sendStatus(200);
        return;
      }

      const embed = new EmbedBuilder()
        .setAuthor({ name: payload.sender?.login || 'GitHub', iconURL: payload.sender?.avatar_url })
        .setColor(0x4caf50)
        .setFooter({ text: payload.repository?.full_name || '' })
        .setTimestamp();

      if (event === 'ping') {
        embed.setTitle(gameData.getMessage('Webhook_Ping_Title')).setDescription(gameData.getMessage('Webhook_Ping_Desc'));
      } else if (event === 'push') {
        if (!payload.commits || !payload.commits.length) {
          res.sendStatus(200);
          return;
        }
        embed.setTitle(gameData.getMessage('Webhook_Push_Title', payload.commits.length));
        const desc = payload.commits
          .slice(0, 5)
          .map((c: any) => `- [\`${c.id.slice(0, 7)}\`](${c.url}) ${c.message}`)
          .join('\n');
        embed.setDescription(desc);
      } else if (event === 'issues') {
        embed
          .setTitle(gameData.getMessage('Webhook_Issue_Title', payload.issue?.number, payload.action))
          .setDescription(gameData.getMessage('Webhook_Issue_Desc', payload.issue?.title, payload.issue?.html_url))
          .setColor(payload.action === 'opened' ? 0xff9800 : 0x4285f4);
      } else {
        console.log(`[Webhook] 처리 안 함(디스코드 미전송): ${String(event)} — push|issues|ping 만 임베드`);
        res.sendStatus(200);
        return;
      }

      for (const channelId of channelIds) {
        const channel = await client.channels.fetch(channelId).catch(() => null);
        if (channel?.isSendable()) {
          await channel.send({ embeds: [embed] }).catch((e: any) =>
            console.error('[Webhook] 채널 전송 실패:', channelId, e?.message ?? e),
          );
        }
      }
      res.sendStatus(200);
    } catch (err: any) {
      console.error('[Webhook] Error:', err?.message ?? err);
      res.sendStatus(500);
    }
  });

  return app;
}

