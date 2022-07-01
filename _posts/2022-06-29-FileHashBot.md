---
title: FileHashBot
date: 2022-06-29 12:10:00 +0200
categories: Project
tags: telegram bot python
image: 
---

🇬🇧 Bot that calculates file hashes.

🇹🇷 Dosya toplamları hesaplayan bot.

Demo in Telegram: [@FileHashBot](https://t.me/FileHashBot)

[![Deploy](https://www.herokucdn.com/deploy/button.svg)](https://heroku.com/deploy?template=https://github.com/HuzunluArtemis/FileHashBot)

🇬🇧 Send media and reply with /hash command. You can clear your all files and quee with /clearme, Your process quee will be cleared. If anything is uploading at now, it will be cleared. Be careful.

🇹🇷 Medyanızı gönderin ve /hash ile yanıtlayın. Tüm dosyalarınızı ve sıranızı /clearme ile temizleyebilirsiniz. İşlem sıranız temizlenir. Şu an yüklenen bir şey varsa temizlenir. Dikkatli olun.

Calculates: `MD5, SHA1, SHA224, SHA256, SHA512, SHA384` If you want more, open pr or issue.

## Setting up config file

Required Variables:
- `BOT_TOKEN`: Telegram Bot Token. Example: `3asd2a2sd32:As56das65d2as:ASd2a6s3d26as`
- `APP_ID`: Telegram App ID. Example: `32523453`
- `API_HASH`: Telegram Api Hash. Example: `asdasdas6d265asd26asd6as1das`
- `AUTH_IDS`: Auth only some groups or users. If you want public, leave it empty or give `0`. Example: `-100656 56191 -10056561`
- `BOT_USERNAME`: Your bot's username. without @. Example: `FileHashBot`

Not Required Variables:
- `OWNER_ID`: Bot's owner id. Send `/id` to `t.me/MissRose_bot` in private to get your id. Required for shell. If you don't want, leave it empty.
- `ONE_PROCESS_PER_USER`: One process per user. Improves bot performance. Example: `0` (False) or `1` (True). Default: `1`
- `FORCE_SUBSCRIBE_CHANNEL`: Force subscribe channel or group. Example: `-1001327202752` or `@HuzunluArtemis`. To disable leave it empty. Do not forget to make admin your bot in forcesub channel or group.
- `CHANNEL_OR_CONTACT`: Your bot's channel or contact username. Example: `HuzunluArtemis`
- `HASH_COMMAND`: Hash command. Default: `hash`
- `STATS_COMMAND`: Server Stats command. Default: `stats`
- `SHELL_COMMAND`: Shell command (only works for owner). Default: `shell`
- `CLEARME_COMMAND`: Clear all user files command. Default: `clearme`
- `DOWNLOAD_DIR`: Downloading directory. Dont change if you dont know about this. Default: `downloads`
- `PROGRESS`: Progress string. See `config.py` or leave it empty.
- `BUFFER_SIZE`: Buffer size for calculate hash. See `config.py` or leave it empty.
- `DOWNLOAD_LIMIT`: File size limit as bytes. See `config.py`. For unlimited give `0`
- `FINISHED_PROGRESS_STR`: Finished Progress Char. Default: `●`
- `UN_FINISHED_PROGRESS_STR`: Unfinished Progress Char. Default: `○`
- `SHOW_PROGRESS_MIN_SIZE_DOWNLOAD`: Progressbar length. Default: `25`
- `UNAUTHORIZED_TEXT_STR`: Unauthorized string. See `config.py`.
- `DOWNLOADING_STR`: Downloading string. See `config.py`.
- `START_TEXT_STR`: Start text string. See `config.py`.
- `CLEAR_STR`: Clearme response string. See `config.py`.
- `ONE_PROCESS_PER_USER_STR`: One process for one user response string. See `config.py`.
- `JOIN_CHANNEL_STR`: Join channel warning string. See `config.py`.
- `YOU_ARE_BANNED_STR`: Banned user string. See `config.py`.
- `JOIN_BUTTON_STR`: Join button string. See `config.py`.

## Lisans

![](https://www.gnu.org/graphics/gplv3-127x51.png)

You can use, study share and improve it at your will. Specifically you can redistribute and/or modify it under the terms of the [GNU General Public License](https://www.gnu.org/licenses/gpl-3.0.html) as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.