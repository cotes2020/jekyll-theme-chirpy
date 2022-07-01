---
title: AutoGroupPrivateChatFilesStoreBot
date: 2022-06-26 12:10:00 +0200
categories: Project
tags: telegram bot python
image: 
---

🇬🇧 This will just save your group files to a Channel & will provide a link to retrieve your file.

🇹🇷 Bu sadece grup dosyalarınızı bir Kanala kaydedecek ve dosyanızı almak için bir bağlantı sağlayacaktır.

[![Deploy](https://www.herokucdn.com/deploy/button.svg)](https://heroku.com/deploy?template=https://github.com/HuzunluArtemis/AutoGroupPrivateChatFilesStoreBot)

🇬🇧 if you forward/send me any media/file, i can create a public link for you.
I will send the file you sent me to anyone who clicks on the link i created.

🇹🇷 bana bir medya/dosya iletir ya da gönderirseniz genel bir bağlantı oluşturabilirim.
oluşturacağım bağlantıya kim tıklarsa ona, bana gönderdiğin dosyayı/medyayı gönderirim.

## Features

- Saves document, video and audio in group. (dont forget to add bot to your group)
- Saves document, video, audio, photo and voice in bot's private chat (if enabled)
- Save Permanently your Group Files
- Save Permanently your Private Files
- Auto delete or not delete saved files
- Ability for saving anonymously to db
- Ability for sending files anonymously to user
- Forcing to Join Channel
- Set custom minimum size for files
- Set custom Blocked extensions
- Set custom url-prefix
- Custom auto-delete time
- Delete message (saved-info) sent by bot with custom time
- Delete file (saved-file) sent by bot with custom time
- Custom auto-kick banned user time
- Custom start message
- Compatible with PublicLeech-like leechers (edit as video)
- Compatible with TorToolkit-like leechers (send new video)
- Send links with bot instead of user. (helps hiding user)
- Auth groups or users or use public
- Run in only bot mode. (Just dont fill user-session)

## Setting up config file

- `BOT_TOKEN`: Telegram Bot Token. Example: `3asd2a2sd32:As56das65d2as:ASd2a6s3d26as`
- `APP_ID`: Telegram App ID. Example: `32523453`
- `API_HASH`: Telegram Api Hash. Example: `asdasdas6d265asd26asd6as1das`
- `STRING_SESSION`: Telegram session string. Example: `3asd2a2sd32:As56das65d2as:ASd2a6s3d26as`
- `DB_CHANNEL_ID`: Files storing channel id. Example: `-10062626626` or `@HuzunluArtemis`
- `FORCE_SUB_CHANNEL`: Force subscribing channel. Example: `-10062626626` or `@HuzunluArtemis`
- `MONGODB_URI`: MongoDB database url. Set ip from everywhere. Example: `mongodb+srv://s:s@xcv.mongodb.net/g?df=true&w=hg`
- `BLOCKED_EXTENSIONS`: This extensions not will be stored. Example: `rar 7z png`
- `BOT_USERNAME`: Your bot's username. without @. Example: `SaverBot`
- `MIN_FILE_SIZE`: For example, if it is 20, files smaller than 20 mb will not be stored. Default is `0`
- `SEND_AS_COPY`: Send as copy to user. Will help for copyright shits. Default: `True`
- `SAVE_AS_COPY`: Save as copy to db. Will help for copyright shits. Default: `True`
- `CONTACT_ADRESS`: Your contacting adress. Example: `@Contactgroup` or `@Contactbot`
- `URL_PREFIX`: URL's prefix. For example for `HA`: your link will be: `t.me/abbot?start=HA_6266`
- `AUTO_DELETE`: Auto-deleting enabled or disabled. Default: `True`
- `AUTO_DELETE_TIME`: Auto-deleting seconds for saved files. Example: `10`
- `AUTO_KICK_TIME`: Auto-kick seconds for banned users. Example: `10`
- `ACCEPT_FROM_PRIVATE`: Accepting backup from bot's private. Example: `False`
- `START_MESSAGE`: Set custom start message. Example: `Bot is running and up.`
- `DELETE_SENT_MESSAGE`: Delete message (saved-info) sent by bot. Example: `True` Default is `False`
- `DELETE_SENT_MESSAGE_TIME`: Set custom seconds for delete "saved-info" messages. Default is `60`
- `DELETE_SENT_FILE`: Delete file (saved-file) sent by bot. Example: `True` Default is `False`
- `DELETE_SENT_FILE_TIME`: Set custom seconds for delete bot's sent files. Default is `60`
- `SKIP_SAVED_INFO_MESSAGE`: Skip saved info message like "Your file saved into db..." Default is `False`
- `USE_BUTTON_FOR_LINK`: Create messages with button. Default is `True`
- `BUTTON_FOR_LINK_STR`: If `USE_BUTTON_FOR_LINK` is true, this value sets the button string. Example: `Click me to get your file`
- `SEND_LINK_AGAIN`: Send link again after sending file. Default is `True`
- `USE_BOT_INSTEAD_USER`: Send links wit bot. (Button will works with groups if this is true.) Default: `True`
- `AUTH_IDS`: Auth only some groups or users. If you want public, leave it empty or give `0`. Example: `-100656 56191 -10056561`
- `ONLY_BOT_MODE`: Run as bot only. Default is `False`

## Thanks

Thanks to original developer: <a href="https://github.com/AbirHasan2005/Save-Group">AbirHasan2005/Save-Group</a> 

## Lisans

![](https://www.gnu.org/graphics/gplv3-127x51.png)

You can use, study share and improve it at your will. Specifically you can redistribute and/or modify it under the terms of the [GNU General Public License](https://www.gnu.org/licenses/gpl-3.0.html) as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.