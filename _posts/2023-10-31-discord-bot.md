---
title: 디스코드 봇으로 깃허브 저장소의 이슈 체크하기 (파이썬사용)
date: 2023-10-21
categories: [troubleshooting]
tags: [python, discord, github, rpa]
---

## 🤔 Problem

전장의 안개 프로젝트를 마무리하는 단계에서, 팀원들이 생계를 꾸리는 동시에 개발을 하니 어려움이 생겼다.
QA 하면서 생긴 이슈들을 데일리 체크하기에 시간과 인력이 모자랐다.

또한 기존에 회사에 다닐 때 슬랙 봇으로 굉장히 많은 업무 자동화를 목격했기에 나도 만들어 보고 싶었다.

그래서 기존에 사용하던 [디스코드 웹훅](https://yubinshin.github.io/posts/github-bot-with-discord/)을 좀 더 발전 시켜 보기로 했다.

결과물로 저장소의 이슈를 asginee 에 따라 매일 트래킹해주는 봇을 만들었다.
![결과물](https://user-images.githubusercontent.com/68121478/279599939-6a059afe-7511-4ff3-8834-b038ee5958cb.png)

## 🌱 Solution

### 디스코드 관련 설정

1. 디스코드 개발자 포털에서 디스코드 봇을 생성한다.

   [디스코드 개발자 포털](https://discord.com/developers/applications)에서 새로운 봇을 생성한다.<br/>
   [이 페이지](https://velog.io/@dombe/문제-해결-그리고-기본적인-Discord-bot-만들기)를 참고하여 기본적인 Discord 봇을 만들어보세요. (intent 설정시 [이 페이지](https://devbench.kr/forum/qna/401)를 참고)<br/>

   > intent 가 생각보다 중요했다.

2. 디스코드 채널 아이디를 구한다.  
   [이 페이지](https://helpdeskgeek.com/how-to/how-to-enable-and-use-developer-mode-on-discord/)를 참고하여 디스코드 채널 ID를 구하세요.<br/>

   ![채널아이디](https://user-images.githubusercontent.com/68121478/279600273-fb195f4d-f19a-45c5-a6f4-38f32431db3b.png)

   > 번외로 이번 프로그램에선 필요없지만 내 개인 아이디의 디스코드 토큰을 얻으려 개발자도구를 열어보니 아래와 같은 메세지가 나와서 귀여웠다ㅋㅋㅋ
   >
   > 이런 식으로 구인하다니 너무 흥미롭잖아!
   >
   > ![번외](https://user-images.githubusercontent.com/68121478/279599139-93f647fc-b3d5-4069-ad3a-ce883ca872f3.png)

### 깃허브 관련 설정

1. 깃허브 Access Token 생성

   [이 페이지](https://hoohaha.tistory.com/37)를 참고하여 깃허브 Access Token을 생성한다.

### discord.py 를 이용해 파이썬 코드 작성

1. 필요한 의존성은 다음과 같다.

   ```sh
   pip install discord.py requests schedule
   ```

2. 파이썬 코드를 작성한다.

   ````py
   import os
   import requests
   import discord
   from discord.ext import commands, tasks
   import time
   from datetime import datetime
   import schedule

   # GitHub API 토큰 및 디스코드 봇 토큰 설정
   # ---------------------------------------------
   GITHUB_TOKEN = "YOUR_GITHUB_TOKEN"
   DISCORD_BOT_TOKEN = "YOUR_DISCORD_BOT_TOKEN"
   DISCORD_CHANNEL_ID = "YOUR_DISCORD_CHANNEL_ID"


   # 디스코드 봇 인스턴스를 생성, Intent 는 디폴트로 설정해주었다.
   # ---------------------------------------------
   client = discord.Client(intents=discord.Intents.default())


   # 매일 깃허브 이슈를 체크하고 디스코드 채널에 메시지를 보내는 함수
   # ---------------------------------------------
   async def send_github_issue_count_once():
      # GitHub API를 사용하여 이슈 가져오기
      headers = {
         "Authorization": f"token {GITHUB_TOKEN}"
      }
      response = requests.get("https://api.github.com/repos/fog-of-war/dev-fe/issues", headers=headers)

      # 응답의 상태코드가 200이라면 assignee 별로 내용을 가공하게 만들었다.
      if response.status_code == 200:
         issues = response.json()
         assignee_count = {}
         issue_messages = []  # 이슈 정보를 담을 리스트
         current_date = datetime.now().strftime("%Y-%m-%d")

         for issue in issues:
               assignee = issue["assignee"]["login"] if issue["assignee"] else "담당자 없음"
               if assignee in assignee_count:
                  assignee_count[assignee] += 1
               else:
                  assignee_count[assignee] = 1

         # 디스코드 채널 ID를 정수로 변환
         channel = client.get_channel(int(DISCORD_CHANNEL_ID))

         # 메시지 생성
         message = f"## 📅 **{current_date}**\n "
         for assignee, count in assignee_count.items():
               message += f"💡 **{assignee}**: {count}개의 이슈가 남아있습니다.\n"
               # message += f"https://github.com/fog-of-war/dev-fe/issues/assigned/{assignee}\n"
               # Collect issue titles within a single code block
               code_block = "```md\n"
               for issue in issues:
                  if issue["assignee"] and issue["assignee"]["login"] == assignee:
                     issue_title = issue["title"]
                     code_block += f"{issue_title}\n"
               code_block += "```"

               message += code_block
               message += "\n"

         # 채널에 가공 완료한 메시지 전송
         await channel.send(message)


   # 매일 오전 9시에 이슈 카운트 함수 실행
   # ---------------------------------------------
   schedule.every().day.at("09:00").do(send_github_issue_count_once)


   # discord.py 에서 제공하는 어노테이션으로 이벤트를 설정한다.
   # ---------------------------------------------
   @client.event
   async def on_ready(): # 인스턴스가 준비 되었을 때
      print(f'Logged in as {client.user}')

      await client.change_presence(status=discord.Status.online, activity=discord.Game("감시"))

   @client.event
   async def on_message(message): # 메세지가 채널에 올라왔을 때 (해당 매세지)
      message_content = message.content
      greet = message_content.find("인사해")
      if greet >= 0:
         await message.channel.send("안녕하세요 감시동균입니다")
      if "누가진짜야" in message_content:
         await message.channel.send("제가 진짜 엉동균입니다")
      if "너누구야" in message_content:
         await message.channel.send("저는 엉동균입니다")
      await client.process_commands(message) # 메세지 중 명령어가 있을 경우 처리해주는 코드


   # 봇 실행
   # ---------------------------------------------
   client.run(DISCORD_BOT_TOKEN)


   # 스케줄링 설정
   # ---------------------------------------------
   while True:
      schedule.run_pending()
      time.sleep(1)  # 스케줄링 루프의 반복 속도를 제어하기 위한 잠시 대기
   ````

### 완성본

[https://github.com/YubinShin/discord-bot.git](https://github.com/YubinShin/discord-bot.git)

1. 저장소를 클론한다.

   ```sh
    git clone https://github.com/YubinShin/discord-bot.git
    cd discord-bot
   ```

2. 가상 환경을 생성&활성화 후 의존성들을 설치한다.

   Mac/linux

   ```sh
   python -m venv myenv
   source myenv/bin/activate
   pip install -r requirements.txt
   ```

   Window

   ```sh
   python -m venv myenv
   myenv\Scripts\activate
   pip install -r requirements.txt
   ```

3. 터미널에 하단 내용을 입력하여 Discord 봇 토큰 및 다른 환경 변수를 설정한다.

   ```sh
   set GITHUB_TOKEN=your_github_token
   set DISCORD_BOT_TOKEN=your_discord_bot_token
   set DISCORD_CHANNEL_ID=your_discord_channel_id
   ```

4. 봇을 실행한다.

   ```sh
   python bot.py
   ```

### 백그라운드 세션에서 파이썬 프로그램 실행

처음엔 깃허브 액션의 워크플로로 실행해보려고 했는데, 실시간으로 계속 접속해있으려면 가상머신에 세션을 켜두는게 좋겠더라.

나는 전장의 안개 ec2 에 켜두었다.

1. 새로운 screen 세션을 시작한다.

   ```sh
   screen -S my_session_name
   ```

2. Python 스크립트를 실행한다.

   ```sh
   python bot.py
   ```

3. 스크린 세션을 종료한다.<br/>
   Ctrl + A, 그 다음, d 키를 누르면 세션을 백그라운드로 보낼 수 있다.<br/>
   스크린 세션을 나중에 다시 찾으려면 "screen -r my_session_name"을 사용한다.

### 봇 사용해보기

- 봇이 실행되면 Discord 서버에 오전 9시마다 깃허브 저장소의 이슈를 트래킹 해줍니다.

  <img width="643" alt="스크린샷 2023-10-31 오후 11 17 01" src="https://github.com/YubinShin/discord-bot/assets/68121478/5ee25f7b-f26b-4646-aefc-5dffac010a16"><br/>

- 봇이 실행되면 Discord 봇이 항상 감시하는 중이라는 표시를 띄워 서버에 긴장감을 줄 수 있다.

  <img width="240" alt="스크린샷 2023-10-31 오후 11 23 42" src="https://github.com/YubinShin/discord-bot/assets/68121478/282570ae-7669-4a9c-89c8-7a5c002fa26e"><br/>

- 봇이 실행되면 Discord 서버에서 다음과 같이 명령어를 입력할 수 있다.

  인사해: 봇이 인사한다. <br/>
  누가진짜야: 봇이 팀원을 사칭한다.<br/>
  너누구야: 봇이 팀원을 사칭한다.<br/>

  <img width="505" alt="스크린샷 2023-10-31 오후 10 44 33" src="https://github.com/YubinShin/discord-bot/assets/68121478/bffe0c13-90cc-4095-a011-721e7f12d16c"><br/>

## 📎 Related articles

| 이슈명                                       | 링크                                                                                                                                                                 |
| -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| discord.py                                   | [https://discordpy-ko.github.io/index.html](https://discordpy-ko.github.io/index.html)                                                                               |
| 디스코드 웹훅                                | [https://yubinshin.github.io/posts/github-bot-with-discord/](https://yubinshin.github.io/posts/github-bot-with-discord/)                                             |
| 디스코드 개발자 포털                         | [https://discord.com/developers/applications](https://discord.com/developers/applications)                                                                           |
| 문제 해결 그리고 기본적인 Discord bot 만들기 | [https://velog.io/@dombe/문제-해결-그리고-기본적인-Discord-bot-만들기](https://velog.io/@dombe/문제-해결-그리고-기본적인-Discord-bot-만들기)                         |
| discord bot intent                           | [https://devbench.kr/forum/qna/401](https://devbench.kr/forum/qna/401)                                                                                               |
| 디스코드 채널 ID 구하는 법                   | [https://helpdeskgeek.com/how-to/how-to-enable-and-use-developer-mode-on-discord/](https://helpdeskgeek.com/how-to/how-to-enable-and-use-developer-mode-on-discord/) |
