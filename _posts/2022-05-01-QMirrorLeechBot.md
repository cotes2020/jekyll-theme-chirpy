---
title: QMirrorLeechBot
date: 2022-05-01 12:10:00 +0200
categories: Project
tags: telegram bot python
image: 
---

This is a Telegram Bot written in Python for mirroring files on the Internet to your Google Drive or Telegram. Based on [python-aria-mirror-bot](https://github.com/lzzy12/python-aria-mirror-bot) and [mirror-leech-telegram-bot](https://github.com/anasty17/mirror-leech-telegram-bot). Written with [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot) telagram api.

## Features: From This Repo
- Fully non-credit repo (For protect repo from ban etc.)
- Some more trackers
- Supports all yandex links like yandex.com.tr, yandex.com.ru
- Better YTDL Playlist naming: `"%(playlist_title)s %(playlist_index)s.%(n_entries)s %(title)s.%(ext)s"`
- Update all python packages before start bot (optional)
- Custom finished & unfinished string. (You can get from [1](https://coolsymbol.com/) [2](https://changaco.oy.lc/unicode-progress-bars/) [3](https://text-symbols.com/) or leave empty for default)
- More respectful to user
- Show thumbnail
- Speedtest with picture
- Removed all username cc's. Never trust a username.
- WayBack Machine Archiver
- VirusTotal File / URL / Hash Scanner (api required)
- Delete file / folder from drive with replying message
- Better file captions. Now writing full path instead of only filename. Before: `Antalya.pdf` After: `Manisa/Antalya.pdf`
- File Hasher: `MD5`, `SHA1`, `SHA224`, `SHA256`, `SHA512`, `SHA384`
- URL Shortener
- MediaInfo (To use, add `RUN apt install mediainfo -y` into your deployer Dockerfile. Do not ask me.)
- Heroku Usage (To use fill `HEROKU_API_KEY` `HEROKU_APP_NAME`)
- Dyno Restarter / Killer (To use fill `HEROKU_API_KEY` `HEROKU_APP_NAME`)
  - Usage: `/restart dyno`. This will update your bot to latest version, re-set your configs with your new values.
  - Usage: `/restart kill`. This will kill your app for save dyno hours.
- AntiSpam Protections:
  - SpamWatch AntiSpam: Fill `SPAMWATCH_ANTISPAM_API` variable.
  - Combot AntiSpam (CAS): Set `COMBOT_CAS_ANTISPAM` to `True`
  - Userge AntiSpam: Fill `USERGE_ANTISPAM_API` variable.
  - IntelliVoid AntiSpam (AI): Set `INTELLIVOID_ANTISPAM` to `True`
- More Shorteners:
  - Not required signup: `v.gd`, `da.gd`, `is.gd`, `ttm.sh`, `clck.ru`, `chilp.it`, `osdb`, `owly`
  - Required signup: `shorte.st`, `bc.vc`, `pubiza`, `linkvertise`, `bit.ly`, `post`, `cutt.ly`, `adf.ly`, `shortcm`, `tinycc`, `tinyurl`, `ouo.io`
  - bc.vc sample api: `"2dgdg5f1fgag7cg6f0622&uid=45634"`
  - pubiza sample api: `"hsdfgCgdgrsdfgsfgfgsdgLsfgXef mainstream"`. Split api key and ad type with a space. Genel içerik için: `mainstream` Yetişkin içerik için: `adult` İçerik Kilidi için: `content_locker`
  - adfly sample api: `"hsdfgCgdgrsdfgsfgfgsdgLsfgXef 51515155"` Split api key and user id with a space.
  - shortcm sample api: `"hsdfgCgdgrsdfgsfgfgsdgLsfgXef abcdotcom"` Split api key and domain with a space.
  - tinycc sample api: `"hsdfgCgdgrsdfgsfgfgsdgLsfgXef tinyccusername"` Split api key and username with a space.
## Features: From Other Repositories
- qBittorrent
- Select files from Torrent before downloading using qbittorrent
- Leech (splitting, thumbnail for each user, setting as document or as media for each user)
- Size limiting for Torrent/Direct, Zip/Unzip, Mega and Clone
- Stop duplicates for all tasks except yt-dlp tasks
- Zip/Unzip G-Drive links
- Counting files/folders from Google Drive link
- View Link button, extra button to open file index link in broswer instead of direct download
- Status Pages for unlimited tasks
- Clone status
- Search in multiple Drive folder/TeamDrive
- Recursive Search (only with `root` or TeamDrive ID, folder ids will be listed with non-recursive method)
- Multi-Search by token.pickle if exists
- Extract rar, zip and 7z splits with or without password
- Zip file/folder with or without password
- Use Token.pickle if file not found with Service Account for all Gdrive functions
- Random Service Account at startup
- Mirror/Leech/Watch/Clone/Count/Del by reply
- YT-DLP quality buttons
- Search for torrents with Torrent Search API or with variable plugins using qBittorrent search engine
- Docker image support for `linux/amd64, linux/arm64, linux/arm/v7, linux/arm/v6` (**Note**: Use `anasty17/mltb-oracle:latest` for oracle or if u faced problem with arm64 docker run)
- Update bot at startup and with restart command using `UPSTREAM_REPO`
- Clone/Zip/Unzip/Count from gdtot links (main script from [Yusuf](https://github.com/oxosec)) and delete first cloned file from main drive or TeamDrive
- Qbittorrent seed until reaching specific ratio or time
- Rss feed and filter. Based on this repository [rss-chan](https://github.com/hyPnOtICDo0g/rss-chan)
- Save leech settings including thumbnails in database
- Many bugs have been fixed
- Mirror direct download links, Torrent, and Telegram files to Google Drive
- Mirror Mega.nz links to Google Drive (If you have non-premium Mega account, it will limit download to 5GB per 6 hours)
- Copy files from someone's Drive to your Drive (Using Autorclone)
- Download/Upload progress, Speeds and ETAs
- Mirror all yt-dlp supported links
- Docker support
- Uploading to Team Drive
- Index Link support
- Service Account support
- Delete files from Drive
- Shortener support
- Speedtest
- Multiple Trackers support
- Shell and Executor
- Sudo with or without Database
- Custom Filename* (Only for direct links, Telegram files and yt-dlp. Not for Mega links, Gdrive links or Torrents)
- Extract or Compress password protected files.
- Extract these filetypes and uploads to Google Drive
  > ZIP, RAR, TAR, 7z, ISO, WIM, CAB, GZIP, BZIP2, APM, ARJ, CHM, CPIO, CramFS, DEB, DMG, FAT, HFS, LZH, LZMA, LZMA2, MBR, MSI, MSLZ, NSIS, NTFS, RPM, SquashFS, UDF, VHD, XAR, Z, tar.xz

- Direct links Supported:
  >letsupload.io, hxfile.co, anonfiles.com, bayfiles.com, antfiles, fembed.com, fembed.net, femax20.com, layarkacaxxi.icu, fcdn.stream, sbplay.org, naniplay.com, naniplay.nanime.in, naniplay.nanime.biz, sbembed.com, streamtape.com, streamsb.net, feurl.com, pixeldrain.com, racaty.net, 1fichier.com, 1drv.ms (Only works for file not folder or business account), uptobox.com (Uptobox account must be premium), solidfiles.com

# How to deploy?

## Prerequisites

- Tutorial Video from A to Z:
  - Thanks to [Wiszky](https://github.com/vishnoe115)
<p><a href="https://www.youtube.com/watch?v=gFQWJ4ftt48"> <img src="https://img.shields.io/badge/See%20Video-black?style=for-the-badge&logo=YouTube" width="160""/></a></p>

### 1. Installing requirements

- Clone this repo.
- For Debian based distros
```
sudo apt install python3
```
Install Docker by following the [official Docker docs](https://docs.docker.com/engine/install/debian/) or by commands below.
```
sudo apt install snapd
sudo snap install docker
```
- For Arch and it's derivatives:
```
sudo pacman -S docker python
```
- Install dependencies for running setup scripts:
```
pip3 install -r requirements-cli.txt
```

------

### 2. Setting up config file

```
cp config_sample.env config.env
```
- Remove the first line saying:
```
_____REMOVE_THIS_LINE_____=True
```
Fill up rest of the fields. Meaning of each field is discussed below:

**1. Required Fields**
<details>
    <summary><b>Click Here For More Details</b></summary>

- `BOT_TOKEN`: The Telegram Bot Token that you got from [@BotFather](https://t.me/BotFather)
- `TELEGRAM_API`: This is to authenticate your Telegram account for downloading Telegram files. You can get this from https://my.telegram.org. **NOTE**: DO NOT put this in quotes.
- `TELEGRAM_HASH`: This is to authenticate your Telegram account for downloading Telegram files. You can get this from https://my.telegram.org
- `OWNER_ID`: The Telegram User ID (not username) of the Owner of the bot.
- `GDRIVE_FOLDER_ID`: This is the Folder/TeamDrive ID of the Google Drive Folder to which you want to upload all the mirrors.
- `DOWNLOAD_DIR`: The path to the local folder where the downloads should be downloaded to.
- `DOWNLOAD_STATUS_UPDATE_INTERVAL`: Time in seconds after which the progress/status message will be updated. Recommended `10` seconds at least.
- `AUTO_DELETE_MESSAGE_DURATION`: Interval of time (in seconds), after which the bot deletes it's message and command message which is expected to be viewed instantly. **NOTE**: Set to `-1` to disable auto message deletion.
- `BASE_URL_OF_BOT`: Valid BASE URL where the bot is deployed to use qbittorrent web selection. Format of URL should be `http://myip`, where `myip` is the IP/Domain(public) of your bot or if you have chosen port other than `80` so write it in this format `http://myip:port` (`http` and not `https`). This Var is optional on VPS and required for Heroku specially to avoid app sleeping/idling. For Heroku fill `https://yourappname.herokuapp.com`. Still got idling? You can use http://cron-job.org to ping your Heroku app.
</details>

**2. Optional Fields**

<details>
    <summary><b>Click Here For More Details</b></summary>

- `ACCOUNTS_ZIP_URL`: Only if you want to load your Service Account externally from an Index Link or by any direct download link NOT webpage link. Archive the accounts folder to ZIP file. Fill this with the direct download link of zip file. If index need authentication so add direct download as shown below:
  - `https://username:password@example.workers.dev/...`
- `TOKEN_PICKLE_URL`: Only if you want to load your **token.pickle** externally from an Index Link. Fill this with the direct link of that file.
- `MULTI_SEARCH_URL`: Check `drive_folder` setup [here](https://github.com/anasty17/mirror-leech-telegram-bot/tree/master#multi-search-ids). Write **drive_folder** file [here](https://gist.github.com/). Open the raw file of that gist, it's URL will be your required variable. Should be in this form after removing commit id: https://gist.githubusercontent.com/username/gist-id/raw/drive_folder
- `YT_COOKIES_URL`: Youtube authentication cookies. Check setup [Here](https://github.com/ytdl-org/youtube-dl#how-do-i-pass-cookies-to-youtube-dl). Use gist raw link and remove commit id from the link, so you can edit it from gists only.
- `NETRC_URL`: To create .netrc file contains authentication for aria2c and yt-dlp. Use gist raw link and remove commit id from the link, so you can edit it from gists only. **NOTE**: After editing .nterc you need to restart the docker or if deployed on heroku so restart dyno in case your edits related to aria2c authentication.
  - **NOTE**: All above url variables used incase you want edit them in future easily without deploying again or if you want to deploy from public fork. If deploying using cli or private fork you can leave these variables empty add token.pickle, accounts folder, drive_folder, .netrc and cookies.txt directly to root but you can't update them without rebuild OR simply leave all above variables and use private UPSTREAM_REPO.
- `DATABASE_URL`: Your Database URL. Follow this [Generate Database](https://github.com/anasty17/mirror-leech-telegram-bot/tree/master#generate-database) to generate database. Data will be saved in Database: auth and sudo users, leech settings including thumbnails for each user and rss data. **NOTE**: If deploying on heroku and using heroku postgresql delete this variable from **config.env** file. **DATABASE_URL** will be grabbed from heroku variables.
- `AUTHORIZED_CHATS`: Fill user_id and chat_id of groups/users you want to authorize. Separate them by space.
- `SUDO_USERS`: Fill user_id of users whom you want to give sudo permission. Separate them by space.
- `IS_TEAM_DRIVE`: Set to `False` or leave it empty to get public google drive links else `True` so only who have access to your Folder/TeamDrive can open the links. `Bool`
- `USE_SERVICE_ACCOUNTS`: Whether to use Service Accounts or not. For this to work see [Using Service Accounts](https://github.com/anasty17/mirror-leech-telegram-bot#generate-service-accounts-what-is-service-account) section below.
- `INDEX_URL`: Refer to https://gitlab.com/ParveenBhadooOfficial/Google-Drive-Index.
- `MEGA_API_KEY`: Mega.nz API key to mirror mega.nz links. Get it from [Mega SDK Page](https://mega.nz/sdk)
- `MEGA_EMAIL_ID`: E-Mail ID used to sign up on mega.nz for using premium account (Leave though)
- `MEGA_PASSWORD`: Password for mega.nz account
- `BLOCK_MEGA_FOLDER`: If you want to remove mega.nz folder support, set it to `True`. `Bool`
- `BLOCK_MEGA_LINKS`: If you want to remove mega.nz mirror support, set it to `True`. `Bool`
- `STOP_DUPLICATE`: (Leave empty if unsure) if this field is set to `True`, bot will check file in Drive, if it is present in Drive, downloading or cloning will be stopped. (**NOTE**: File will be checked using filename not file hash, so this feature is not perfect yet). `Bool`
- `CLONE_LIMIT`: To limit the size of Google Drive folder/file which you can clone. Don't add unit, the default unit is `GB`.
- `MEGA_LIMIT`: To limit the size of Mega download. Don't add unit, the default unit is `GB`.
- `TORRENT_DIRECT_LIMIT`: To limit the Torrent/Direct mirror size. Don't add unit, the default unit is `GB`.
- `ZIP_UNZIP_LIMIT`: To limit the size of zip and unzip commands. Don't add unit, the default unit is `GB`.
- `VIEW_LINK`: View Link button to open file Index Link in browser instead of direct download link, you can figure out if it's compatible with your Index code or not, open any video from you Index and check if its URL ends with `?a=view`, if yes make it `True`, compatible with [BhadooIndex](https://gitlab.com/ParveenBhadooOfficial/Google-Drive-Index) Code. `Bool`
- `UPTOBOX_TOKEN`: Uptobox token to mirror uptobox links. Get it from [Uptobox Premium Account](https://uptobox.com/my_account).
- `IGNORE_PENDING_REQUESTS`: If you want the bot to ignore pending requests after it restarts, set this to `True`. `Bool`
- `STATUS_LIMIT`: Limit the no. of tasks shown in status message with buttons. **NOTE**: Recommended limit is `4` tasks.
- `IS_VPS`: (Only for VPS) Don't set this to `True` even if you are using VPS, unless facing error with web server. `Bool`
- `SERVER_PORT`: Only For VPS even if `IS_VPS` is `False`, which is the **BASE_URL_OF_BOT** Port.
- `WEB_PINCODE`: If empty or `False` means no more pincode required while qbit web selection. `Bool`
- `QB_SEED`: If `True` QB torrent will be seeded after and while uploading until reaching specific ratio or time, edit `MaxRatio` or `GlobalMaxSeedingMinutes` or both from qbittorrent.conf (`-1` means no limit, but u can cancel manually by gid). **NOTE**: 1. Don't change `MaxRatioAction`, 2. Only works with `/qbmirror` and `/qbzipmirror`. `Bool`
- `QB_TIMEOUT`: Timeout of dead torrents downloading with qBittorrent in seconds.
- `TG_SPLIT_SIZE`: Size of split in bytes, leave it empty for max size `2GB`.
- `AS_DOCUMENT`: Default Telegram file type upload. Empty or `False` means as media. `Bool`
- `EQUAL_SPLITS`: Split files larger than **TG_SPLIT_SIZE** into equal parts size (Not working with zip cmd). `Bool`
- `CUSTOM_FILENAME`: Add custom word to leeched file name.
- `UPSTREAM_REPO`: Your github repository link, if your repo is private add `https://username:{githubtoken}@github.com/{username}/{reponame}` format. Get token from [Github settings](https://github.com/settings/tokens). So you can update your appllication from filled repository on each restart. **NOTE**: Any change in docker or requirements you need to deploy/build again with updated repo to take effect - DON'T delete .gitignore file.
- `SHORTENER_API`: Fill your Shortener API key.
- `SHORTENER`: Shortener URL.
  - Supported URL Shorteners:
  >exe.io, gplinks.in, shrinkme.io, urlshortx.com, shortzon.com, bit.ly, shorte.st, linkvertise.com , ouo.io
- `SEARCH_API_LINK`: Search api app link. Get your api from deploying this [repository](https://github.com/Ryuk-me/Torrents-Api).
  - Supported Sites:
  >1337x, YTS, Eztv, Torrent Galaxy, Torlock, Piratebay, Nyaasi, Rarbg, Ettv, Zooqle, KickAss, Bitsearch, Glodls, MagnetDL, TorrentProject, TorrentFunk, LimeTorrent
- `PHPSESSID` and `CRYPT`: Cookies for gdtot google drive link generator. Follow these [steps](https://github.com/anasty17/mirror-leech-telegram-bot/tree/master#gdtot-cookies).
- `SEARCH_PLUGINS`: List of qBittorrent search plugins (github raw links). I have added some plugins, you can remove/add plugins as you want. Main Source: [qBittorrent Search Plugins (Official/Unofficial)](https://github.com/qbittorrent/search-plugins/wiki/Unofficial-search-plugins).
- `RSS_DELAY`: Time in seconds for rss refresh interval. Recommended `900` seconds at least. Empty means 900 s (default time).
- `RSS_COMMAND`: Choose command for the desired action.
- `RSS_CHAT_ID`: Chat ID where bot will send the rss links.
- `USER_STRING_SESSION`: To send rss links from your telegram account instead of adding bot to channel then adding channel to group to get rss link since bot will not read command from itself or other bot. To generate string session use this command `python3 generate_string_session.py` after mounting repo folder for sure.
  - **RSS NOTE**: `DATABASE_URL` and `RSS_CHAT_ID` is required, otherwise all rss commands will not work.
- Three buttons are already added including Drive Link, Index Link, and View Link, you can add extra buttons, if you don't know what are the below entries, simply leave them empty.
  - `BUTTON_FOUR_NAME`:
  - `BUTTON_FOUR_URL`:
  - `BUTTON_FIVE_NAME`:
  - `BUTTON_FIVE_URL`:
  - `BUTTON_SIX_NAME`:
  - `BUTTON_SIX_URL`:

</details>

------

### 3. Getting Google OAuth API credential file and token.pickle
- Visit the [Google Cloud Console](https://console.developers.google.com/apis/credentials)
- Go to the OAuth Consent tab, fill it, and save.
- Go to the Credentials tab and click Create Credentials -> OAuth Client ID
- Choose Desktop and Create.
- Publish your OAuth consent screen App to prevent **token.pickle** from expire
- Use the download button to download your credentials.
- Move that file to the root of mirrorbot, and rename it to **credentials.json**
- Visit [Google API page](https://console.developers.google.com/apis/library)
- Search for Drive and enable it if it is disabled
- Finally, run the script to generate **token.pickle** file for Google Drive:
```
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
python3 generate_drive_token.py
```
------

## Deploying on VPS

**IMPORTANT NOTES**:
1. You must set `SERVER_PORT` variable to `80` or any other port you want to use.
2. To clear the container (this will not affect on the image):
```
sudo docker container prune
```
3. To delete the images:
```
sudo docker image prune -a
```
4. Check the number of processing units of your machine with `nproc` cmd and times it by 4, then edit `AsyncIOThreadsCount` in qBittorrent.conf.
5. Use `anasty17/mltb-oracle:latest` for oracle or if u faced problem with arm64 docker run.
   - Tutorial Video for Deploying on Oracle VPS:
     - Thanks to [Wiszky](https://github.com/vishnoe115)
     - No need to use sudo su, you can also use sudo before each cmd!
<p><a href="https://youtu.be/IzUG7U7v4U4?t=968"> <img src="https://img.shields.io/badge/See%20Video-black?style=for-the-badge&logo=YouTube" width="160""/></a></p>

------

### Deploying on VPS Using Docker

- Start Docker daemon (skip if already running):
```
sudo dockerd
```
- **Note**: If not started or not starting, run the command below then try to start.
```
sudo apt install docker.io
```
- Build Docker image:
```
sudo docker build . -t mirror-bot
```
- Run the image:
```
sudo docker run -p 80:80 mirror-bot
```
- To stop the image:
```
sudo docker ps
```
```
sudo docker stop id
```

----

### Deploying on VPS Using docker-compose

**NOTE**: If you want to use port other than 80, change it in [docker-compose.yml](https://github.com/anasty17/mirror-leech-telegram-bot/blob/master/docker-compose.yml) also.

```
sudo apt install docker-compose
```
- Build and run Docker image:
```
sudo docker-compose up
```
- After editing files with nano for example (nano start.sh):
```
sudo docker-compose up --build
```
- To stop the image:
```
sudo docker-compose stop
```
- To run the image:
```
sudo docker-compose start
```
- Tutorial video from Tortoolkit repo for docker-compose and checking ports
<p><a href="https://youtu.be/c8_TU1sPK08"> <img src="https://img.shields.io/badge/See%20Video-black?style=for-the-badge&logo=YouTube" width="160""/></a></p>

------

## Deploying on Heroku
<p><a href="https://github.com/anasty17/mirror-leech-telegram-bot/tree/heroku"> <img src="https://img.shields.io/badge/Deploy%20Guide-blueviolet?style=for-the-badge&logo=heroku" width="170""/></a></p>

------

# Extras

## Bot commands to be set in [@BotFather](https://t.me/BotFather)

```
mirror - Mirror
zipmirror - Mirror and upload as zip
unzipmirror - Mirror and extract files
qbmirror - Mirror torrent using qBittorrent
qbzipmirror - Mirror torrent and upload as zip using qb
qbunzipmirror - Mirror torrent and extract files using qb
leech - Leech
zipleech - Leech and upload as zip
unzipleech - Leech and extract files
qbleech - Leech torrent using qBittorrent
qbzipleech - Leech torrent and upload as zip using qb
qbunzipleech - Leech torrent and extract using qb
clone - Copy file/folder to Drive
count - Count file/folder of Drive
watch - Mirror yt-dlp supported link
zipwatch - Mirror yt-dlp supported link as zip
leechwatch - Leech through yt-dlp supported link
leechzipwatch - Leech yt-dlp support link as zip
leechset - Leech settings
setthumb - Set thumbnail
status - Get Mirror Status message
rsslist - List all subscribed rss feed info
rssget - Get specific No. of links from specific rss feed
rsssub - Subscribe new rss feed
rssunsub - Unsubscribe rss feed by title
rssunsuball - Remove all rss feed subscriptions
list - Search files in Drive
search - Search for torrents with API
cancel - Cancel a task
cancelall - Cancel all tasks
del - Delete file/folder from Drive
log - Get the Bot Log
shell - Run commands in Shell
restart - Restart the Bot
stats - Bot Usage Stats
ping - Ping the Bot
wayback - Archive a webpage
virustotal - Scan a file or url
speedtest - Speed Test
hash - Calculate hash MD5, SHA1
short - URL Shortener
mediainfo - Extract Media Info
help - All cmds with description
```
------

## Using Service Accounts for uploading to avoid user rate limit
>For Service Account to work, you must set `USE_SERVICE_ACCOUNTS` = "True" in config file or environment variables.
>**NOTE**: Using Service Accounts is only recommended while uploading to a Team Drive.

### 1. Generate Service Accounts. [What is Service Account?](https://cloud.google.com/iam/docs/service-accounts)
Let us create only the Service Accounts that we need.

**Warning**: Abuse of this feature is not the aim of this project and we do **NOT** recommend that you make a lot of projects, just one project and 100 SAs allow you plenty of use, its also possible that over abuse might get your projects banned by Google.

>**NOTE**: If you have created SAs in past from this script, you can also just re download the keys by running:
```
python3 gen_sa_accounts.py --download-keys project_id
```
>**NOTE:** 1 Service Account can upload/copy around 750 GB a day, 1 project can make 100 Service Accounts so you can upload 75 TB a day or clone 2 TB from each file creator (uploader email).

#### Two methods to create service accounts
Choose one of these methods

##### 1. Create Service Accounts in existed Project (Recommended Method)
- List your projects ids
```
python3 gen_sa_accounts.py --list-projects
```
- Enable services automatically by this command
```
python3 gen_sa_accounts.py --enable-services $PROJECTID
```
- Create Sevice Accounts to current project
```
python3 gen_sa_accounts.py --create-sas $PROJECTID
```
- Download Sevice Accounts as accounts folder
```
python3 gen_sa_accounts.py --download-keys $PROJECTID
```

##### 2. Create Service Accounts in New Project
```
python3 gen_sa_accounts.py --quick-setup 1 --new-only
```
A folder named accounts will be created which will contain keys for the Service Accounts.

### 2. Add Service Accounts

#### Two methods to add service accounts
Choose one of these methods

##### 1. Add Them To Google Group then to Team Drive (Recommended)
- Mount accounts folder
```
cd accounts
```
- Grab emails form all accounts to emails.txt file that would be created in accounts folder
- `For Windows using PowerShell`
```
$emails = Get-ChildItem .\**.json |Get-Content -Raw |ConvertFrom-Json |Select -ExpandProperty client_email >>emails.txt
```
- `For Linux / MacOs`
```
grep -oPh '"client_email": "\K[^"]+' *.json > emails.txt
```
- Unmount acounts folder
```
cd ..
```
Then add emails from emails.txt to Google Group, after that add this Google Group to your Shared Drive and promote it to manager and delete email.txt file from accounts folder

##### 2. Add Them To Team Drive Directly
- Run:
```
python3 add_to_team_drive.py -d SharedTeamDriveSrcID
```
------

### Generate Database

**1. Using Railway**
- Go to [railway](https://railway.app) and create account
- Start new project
- Press on `Provision PostgreSQL`
- After creating database press on `PostgresSQL`
- Go to `Connect` column
- Copy `Postgres Connection URL` and fill `DATABASE_URL` variable with it

**2. Using Heroku PostgreSQL**
<p><a href="https://dev.to/prisma/how-to-setup-a-free-postgresql-database-on-heroku-1dc1"> <img src="https://img.shields.io/badge/See%20Dev.to-black?style=for-the-badge&logo=dev.to" width="160""/></a></p>

**3. Using ElephantSQL**
- Go to [elephantsql](https://elephantsql.com) and create account
- Hit `Create New Instance`
- Follow the further instructions in the screen
- Hit `Select Region`
- Hit `Review`
- Hit `Create instance`
- Select your database name
- Copy your database url, and fill `DATABASE_URL` variable with it

------

## Multi Search IDs
To use list from multi TD/folder. Run driveid.py in your terminal and follow it. It will generate **drive_folder** file or u can simply create `drive_folder` file in working directory and fill it, check below format:
```
MyTdName folderID/tdID IndexLink(if available)
MyTdName2 folderID/tdID IndexLink(if available)
```
-----

## Yt-dlp and Aria2c Authentication Using .netrc File
For using your premium accounts in yt-dlp or for protected Index Links, create .netrc file according to following format:

**Note**: Create .netrc and not netrc, this file will be hidden, so view hidden files to edit it after creation.

Format:
```
machine host login username password my_password
```
Example:
```
machine instagram login anas.tayyar password mypassword
```
**Instagram Note**: You must login even if you want to download public posts and after first try you must confirm that this was you logged in from different ip(you can confirm from phone app).

**Youtube Note**: For `youtube` authentication use [cookies.txt](https://github.com/ytdl-org/youtube-dl#how-do-i-pass-cookies-to-youtube-dl) file.

For Index Link with only password without username, even http auth will not work, so this is the solution.
```
machine example.workers.dev password index_password
```
Where host is the name of extractor (eg. instagram, Twitch). Multiple accounts of different hosts can be added each separated by a new line.

-----

## Gdtot Cookies
To Clone or Leech gdtot link follow these steps:
1. Login/Register to [gdtot](https://new.gdtot.top).
2. Copy this script and paste it in browser address bar.
   - **Note**: After pasting it check at the beginning of the script in broswer address bar if `javascript:` exists or not, if not so write it as shown below.
   ```
   javascript:(function () {
     const input = document.createElement('input');
     input.value = JSON.stringify({url : window.location.href, cookie : document.cookie});
     document.body.appendChild(input);
     input.focus();
     input.select();
     var result = document.execCommand('copy');
     document.body.removeChild(input);
     if(result)
       alert('Cookie copied to clipboard');
     else
       prompt('Failed to copy cookie. Manually copy below cookie\n\n', input.value);
   })();
   ```
   - After pressing enter your browser will prompt a alert.
3. Now you'll get this type of data in your clipboard
   ```
   {"url":"https://new.gdtot.org/","cookie":"PHPSESSID=k2xxxxxxxxxxxxxxxxxxxxj63o; crypt=NGxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxWdSVT0%3D"}

   ```
4. From this you have to paste value of PHPSESSID and crypt in config.env file.

-----

## Lisans

![](https://www.gnu.org/graphics/gplv3-127x51.png)

You can use, study share and improve it at your will. Specifically you can redistribute and/or modify it under the terms of the [GNU General Public License](https://www.gnu.org/licenses/gpl-3.0.html) as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.