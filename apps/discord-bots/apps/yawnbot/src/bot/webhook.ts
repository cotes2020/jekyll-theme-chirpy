import express from 'express';
import { EmbedBuilder } from 'discord.js';
import type { Client } from 'discord.js';
import type { GameDataService } from '../services/gamedata';
import { getChannelsForRepo } from '../services/webhook-routes';

export function createGithubWebhookApp(client: Client, gameData: GameDataService) {
  const app = express();
  app.use(express.json());

  app.post('/webhook/github', async (req, res) => {
    try {
      const event = req.headers['x-github-event'];
      const payload = req.body;
      console.log(`[Webhook] Received: ${event}`);

      const repoFullName: string | undefined = payload.repository?.full_name;
      const channelIds = getChannelsForRepo(repoFullName);
      if (channelIds.length === 0) {
        console.warn(
          `[Webhook] ${repoFullName ?? '?'} 매칭 채널 없음 — 디스코드 전송 생략 (data/webhook-routes.json 확인)`,
        );
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
      } else if (event === 'pull_request') {
        const pr = payload.pull_request;
        if (!pr) {
          res.sendStatus(200);
          return;
        }
        let action: string = payload.action;
        if (action === 'closed' && pr.merged) action = 'merged';

        const colorByAction: Record<string, number> = {
          opened: 0x2cbe4e,
          reopened: 0xff9800,
          merged: 0x6f42c1,
          closed: 0xcb2431,
          ready_for_review: 0x4285f4,
        };
        if (!(action in colorByAction)) {
          console.log(`[Webhook] PR action 무시: ${action}`);
          res.sendStatus(200);
          return;
        }

        embed
          .setTitle(gameData.getMessage('Webhook_PR_Title', pr.number, action))
          .setDescription(
            gameData.getMessage(
              'Webhook_PR_Desc',
              pr.title,
              pr.html_url,
              pr.head?.ref ?? '?',
              pr.base?.ref ?? '?',
            ),
          )
          .setColor(colorByAction[action]);
      } else if (event === 'release') {
        const release = payload.release;
        if (!release || payload.action !== 'published') {
          console.log(`[Webhook] Release action 무시: ${payload.action}`);
          res.sendStatus(200);
          return;
        }
        embed
          .setTitle(gameData.getMessage('Webhook_Release_Title', release.tag_name))
          .setDescription(
            gameData.getMessage(
              'Webhook_Release_Desc',
              release.name || release.tag_name,
              release.html_url,
            ),
          )
          .setColor(release.prerelease ? 0xff9800 : 0x6f42c1);
      } else {
        console.log(`[Webhook] 처리 안 함(디스코드 미전송): ${String(event)} — push|issues|pull_request|release|ping 만 임베드`);
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

