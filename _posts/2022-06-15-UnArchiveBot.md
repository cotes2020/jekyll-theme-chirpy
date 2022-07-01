---
title: UnArchiveBot
date: 2022-06-15 12:10:00 +0200
categories: Project
tags: telegram bot python
image: 
---

🇬🇧 Bot that allows you to extract supported archive formats in telegram.

🇹🇷 Desteklenen arşiv biçimleri telegram içerisinde çıkarmanızı sağlayan bot.

Demo in Telegram: [@UnArchiveBot](https://t.me/UnArchiveBot)

[![Deploy](https://www.herokucdn.com/deploy/button.svg)](https://heroku.com/deploy?template=https://github.com/HuzunluArtemis/UnArchiveBot)

🇬🇧 Send archive and reply with /unzip (if passworded: Leave a space after the command and enter the password.) You can clear your all files with /clearme, Your process quee will be cleared. If anything is uploading at now, it will be cleared. Be careful.

🇹🇷 Arşivi gönderin ve /unzip ile yanıtlayın. (parolalıysa: komutunuzdan sonra bir boşluk bırakıp parolayı girin.) Tüm dosyalarınızı /clearme ile temizleyebilirsiniz. İşlem sıranız temizlenir. Şu an yüklenen bir şey varsa temizlenir. Dikkatli olun.

🍓 Örnekler / Samples:

✅ /unzip

✅ /unzip Hunhj887ZunLudArt87emiS

✅ /unzip HEreis8yorupassword-parolaizZBuraya

✅ /unzip anoTherSampLe-bAskABirorNek

🖼 Set thumbnail / Küçük resim ayarlama: /save

❌ Clear thumbnail / Küçük resmi temizle: /clear

🌆 Show thumbnail / Küçük resmi göster: /show

🌿 Server stats / Sunucu istatistikleri: /stats

## Features

- Supports: `7Z, APM, ARJ, BZ2, BZIP2, CAB, CHM, CPIO, CRAMFS, DEB, DMG, FAT, GZ, GZIP, HFS, ISO, LZH, LZMA, LZMA2, MBR, MSI, MSLZ, NSIS, NTFS, RAR, RPM, SQUASHFS, TAR, TAR.BZ2, TAR.GZ, TAR.XZ, TBZ2, TGZ, UDF, VHD, WIM, XAR, Z, ZIP`
- Extract with password
- Send all in document format (Not dynamic at now. Will be selectable while extracting in future)
- Send with thumbnail (not permanent. Check with your check thumbnail command every time.)
- Clear all files from server
- Natural sorting while sending files. More info: `https://en.wikipedia.org/wiki/Natural_sort_order`
- Owner shell
- Force users to join channel or group
- Auth only for some users or make public
- One process per user (for bot performance)
- Changeable upload / download texts, commands, contact adress etc.
- Disable extensions
- Change progressbar (length, unfinished / finished char)
- Custom sleep time between sending files. (recommended: `2`)

## Setting up config file

Required Variables:
- `BOT_TOKEN`: Telegram Bot Token. Example: `3asd2a2sd32:As56das65d2as:ASd2a6s3d26as`
- `APP_ID`: Telegram App ID. Example: `32523453`
- `API_HASH`: Telegram Api Hash. Example: `asdasdas6d265asd26asd6as1das`
- `AUTH_IDS`: Auth only some groups or users. If you want public, leave it empty or give `0`. Example: `-100656 56191 -10056561`
- `BOT_USERNAME`: Your bot's username. without @. Example: `UnArchiveBot`
- `OWNER_ID`: Bot's owner id. Send `/id` to `t.me/MissRose_bot` in private to get your id.

Not Required Variables:
- `ONE_PROCESS_PER_USER`: One process per user. Improves bot performance. Example: `0` (False) or `1` (True). Default: `1`
- `FORCE_SUBSCRIBE_CHANNEL`: Force subscribe channel or group. Example: `-1001327202752` or `@HuzunluArtemis`. To disable leave it empty. Do not forget to make admin your bot in forcesub channel or group.
- `FORCE_DOC_UPLOAD`: Force send all files as document. Without compress videos, photos etc. Example: `0` (False) or `1` (True). Default: `0`
- `CHANNEL_OR_CONTACT`: Your bot's channel or contact username. Example: `HuzunluArtemis`
- `SLEEP_TIME_BETWEEN_SEND_FILES`: Sleep time between files. For floodwait. Recommended: `2`
- `EXTENSIONS`: Supported extensions in bot. To disable, delete from full default list and fill. See `config.py`.
- `UNZIP_COMMAND`: Unzip command. Default: `unzip`
- `STATS_COMMAND`: Server Stats command. Default: `stats`
- `SHELL_COMMAND`: Shell command (only works for owner). Default: `shell`
- `CLEARME_COMMAND`: Clear all user files command. Default: `clearme`
- `SAVE_THUMB_COMMAND`: Save thumbnail command. Reply to a photo to save. Default: `save`
- `CLEAR_THUMB_COMMAND`: Clear thumbnail command. Default: `clear`
- `SHOW_THUMB_COMMAND`: Show user thumbnail command. Default: `show`
- `SORT_FILES_BEFORE_SEND`: Sort all files and send. Example: `0` (False) or `1` (True). Default: `1`
- `USE_NATSORT`: Use natural sort instead of alphabetical sort. Example: `0` (False) or `1` (True). Default: `1`
- `DOWNLOAD_DIR`: Downloading directory. Dont change if you dont know about this. Default: `downloads`
- `PROGRESS`: Progress string with 6 variables. See `config.py`.
- `FINISHED_PROGRESS_STR`: Finished Progress Char. Default: `●`
- `UN_FINISHED_PROGRESS_STR`: Unfinished Progress Char. Default: `○`
- `SHOW_PROGRESS_MIN_SIZE_DOWNLOAD`: Progressbar length. Default: `25`
- `UNAUTHORIZED_TEXT_STR`: Unauthorized string. See `config.py`.
- `DOWNLOADING_STR`: Downloading string. See `config.py`.
- `UPLOADING_STR`: Uploading string. See `config.py`.
- `START_TEXT_STR`: Start text string. See `config.py`.
- `UPLOAD_SUCCESS`: Upload success string. See `config.py`.
- `CLEAR_STR`: Clearme response string. See `config.py`.
- `ONE_PROCESS_PER_USER_STR`: One process for one user response string. See `config.py`.
- `JOIN_CHANNEL_STR`: Join channel warning string. See `config.py`.
- `YOU_ARE_BANNED_STR`: Banned user string. See `config.py`.
- `JOIN_BUTTON_STR`: Join button string. See `config.py`.

## Lisans

![](https://www.gnu.org/graphics/gplv3-127x51.png)

You can use, study share and improve it at your will. Specifically you can redistribute and/or modify it under the terms of the [GNU General Public License](https://www.gnu.org/licenses/gpl-3.0.html) as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.