---
title: WaybackBot
date: 2022-06-17 12:10:00 +0200
categories: Project
tags: telegram bot python
image: 
---

🇬🇧 If you send me a link i will wayback it for you.

🇹🇷 Bana bir link gönderirsen onu arşivlemeye çalışacağım.

Demo in Telegram: [@ArchiveOrgBot](https://t.me/ArchiveOrgBot)

[![Deploy](https://www.herokucdn.com/deploy/button.svg)](https://heroku.com/deploy?template=https://github.com/HuzunluArtemis/WaybackBot)

## Setting up config file

- `BOT_TOKEN`: Telegram Bot Token. Example: `3asd2a2sd32:As56das65d2as:ASd2a6s3d26as`
- `APP_ID`: Telegram App ID. Example: `32523453`
- `API_HASH`: Telegram Api Hash. Example: `asdasdas6d265asd26asd6as1das`
- `AUTH_IDS`: Auth only some groups or users. If you want public, leave it empty or give `0`. Example: `-100656 56191 -10056561`
- `BOT_USERNAME`: Your bot's username. without @. Example: `ArchiveOrgBot`

<b>Not Required Variables:</b>

- `OWNER_ID`: Bot's owner id. Send `/id` to `t.me/MissRose_bot` in private to get your id. Required for shell. If you don't want, leave it empty.
- `FORCE_SUBSCRIBE_CHANNEL`: Force subscribe channel or group. Example: `-1001327202752` or `@HuzunluArtemis`. To disable leave it empty. Do not forget to make admin your bot in forcesub channel or group.
- `CHANNEL_OR_CONTACT`: Your bot's channel or contact username. Example: `HuzunluArtemis`
- `JOIN_CHANNEL_STR`: Join channel warning string. See `config.py`.
- `AUTO_SAVE_ALL_URLS`: Save all urls instantly. Default: `False`.
- `YOU_ARE_BANNED_STR`: Banned user string. See `config.py`.
- `JOIN_BUTTON_STR`: Join button string. See `config.py`.

## Lisans

![](https://www.gnu.org/graphics/gplv3-127x51.png)

You can use, study share and improve it at your will. Specifically you can redistribute and/or modify it under the terms of the [GNU General Public License](https://www.gnu.org/licenses/gpl-3.0.html) as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.