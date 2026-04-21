---
title: "이세돌 찾기"
# description: ""
categories: [작업물, 버추얼]
tags: [작업물, VRChat, 유니티]
image: "/assets/img/background/20230112-151539.jpg"
hidden: true

date: 2024-10-19. 09:59
last_modified_at: 2024-10-21. 10:14 # Init
---

## 기획

---

- 아트 작업자 힉민님의 SVN

- 칸막이 칸마다 사운드 개별
- 탈락자 대기실로 소환 기능
- 채팅 UI
- 공지 채팅
- 투표
- 투표를 위한 -> 채팅방 분리
- 닉네임 변경
- 텔레포트
- 우승 연출

- 채팅 분리가 되어야 함
- 데이터를 DataDictionary로
  - 시간
  - 보낸사람
  - 메시지
  - 추가정보
    - \+ 가상 닉네임
    - \+ 공지 유무? 채팅 타입?
    - \+ 채팅창 위치

// 원래는 이런식

```cs
using System.Text;
using TMPro;
using UdonSharp;
using UnityEngine;
using UnityEngine.UI;
using VRC.SDKBase;
using VRC.Udon;
using WRC.Woodon;

namespace Mascari4615.Project
{
    public class ChattingManager: MBase
    {
        [Header("_" + nameof(ChattingManager))]
        [SerializeField] private MPlayerUdonIndex mPlayerUdonIndex;
        // [SerializeField] private KoreanKeyboard koreanKeyboard;
        [SerializeField] private MString[] chatJSONs;
        [SerializeField] private TextMeshProUGUI chatText;
        // [SerializeField] private TMP_InputField chatInputField;
        [SerializeField] private InputField chatInputField;

        [SerializeField] private MSFXManager mSFXManager;

        private const char SPACE_CHAR = '@';

        private void Start()
        {
            Init();
        }

        private void Init()
        {
            chatText.text = string.Empty;
            for (int i = 0; i < chatJSONs.Length; i++)
                chatJSONs[i].RegisterListener(this, nameof(ReceieveChat) + i);
        }

        #region SendChat
        public void SendChatMessage(string message)
        {
            MDebugLog($"{nameof(SendChatMessage)}: {message}");

            if (string.IsNullOrEmpty(message))
                return;

            message = message.TrimEnd();

            // 클라이언트가 호출
            int udonIndex = mPlayerUdonIndex.GetUdonIndex();
            if (udonIndex == NONE_INT)
            {
                MDebugLog("Udon Index is None.");
                return;
            }

            string time = Networking.GetServerTimeInMilliseconds().ToString();

            MString mString = chatJSONs[udonIndex];
            mString.SetValue($"{time}{SPACE_CHAR}{Networking.LocalPlayer.displayName}{SPACE_CHAR}{message}");
        }

        public void SendChatMessage_KoreanKeyboard()
        {
            MDebugLog($"{nameof(SendChatMessage_KoreanKeyboard)}");

            if (string.IsNullOrEmpty(chatInputField.text))
                return;

            SendChatMessage(chatInputField.text);
            chatInputField.text = string.Empty;

            // SendChatMessage(koreanKeyboard.CurString);
        }
        #endregion

        #region ReceieveChat
        public void ReceieveChat(int udonIndex)
        {
            MDebugLog($"{nameof(ReceieveChat)}: {udonIndex}");

            MString mString = chatJSONs[udonIndex];
            string[] strings = mString.Value.Split(SPACE_CHAR);

            if (strings == null || strings.Length != 3)
            {
                MDebugLog("Strings is Null or Length is not 3.");
                return;
            }

            int time = int.Parse(strings[0]);
            string name = strings[1];
            string message = strings[2];

            ReceieveChat(time, name, message, udonIndex);

            if (udonIndex == mPlayerUdonIndex.GetUdonIndex())
            {
                // chatInputField.Select();
                // chatInputField.ActivateInputField();
            }

            mSFXManager.PlaySFX_L(0);
        }

        public void ReceieveChat(int time, string name, string message, int udonIndex)
        {
            string allText = chatText.text.TrimEnd();
            string[] strings = allText.Split('\n');

            int[] times = new int[strings.Length + 1];
            string[] chats = new string[strings.Length + 1];
            {
                times[0] = time;
                chats[0] = $"{udonIndex}-{name} :: {message}";
            }

            for (int i = 0; i < strings.Length; i++)
            {
                string[] split = strings[i].Split("\t\t| ");
                if (split == null || split.Length != 2)
                {
                    MDebugLog("Split is Null or Length is not 2.");
                    continue;
                }

                int targetIndex = i + 1;
                times[targetIndex] = int.Parse(split[0]);
                chats[targetIndex] = split[1];
                // MDebugLog($"{strings[i]} => {times[targetIndex]} | {chats[targetIndex]}");
            }

            // Sort by Time (오름차순)
            for (int i = 0; i < times.Length; i++)
            {
                for (int j = i + 1; j < times.Length; j++)
                {
                    if (times[i] > times[j])
                    {
                        int tempTime = times[i];
                        times[i] = times[j];
                        times[j] = tempTime;

                        string tempChat = chats[i];
                        chats[i] = chats[j];
                        chats[j] = tempChat;
                    }
                }
            }

            StringBuilder stringBuilder = new StringBuilder();
            int chatCount = 0;
            for (int i = times.Length - 1; i >= 0; i--)
            {
                if (times[i] == 0)
                    continue;
                
                stringBuilder.Insert(0, $"{times[i]}\t\t| {chats[i]}\n");
                chatCount++;

                if (chatCount >= 10)
                    break;
            }
            chatText.text = stringBuilder.ToString();
        }
        #endregion

        #region HorribleEvents
        public void ReceieveChat0() => ReceieveChat(0);
        public void ReceieveChat1() => ReceieveChat(1);
        public void ReceieveChat2() => ReceieveChat(2);
        public void ReceieveChat3() => ReceieveChat(3);
        public void ReceieveChat4() => ReceieveChat(4);
        public void ReceieveChat5() => ReceieveChat(5);
        public void ReceieveChat6() => ReceieveChat(6);
        public void ReceieveChat7() => ReceieveChat(7);
        public void ReceieveChat8() => ReceieveChat(8);
        public void ReceieveChat9() => ReceieveChat(9);
        public void ReceieveChat10() => ReceieveChat(10);
        public void ReceieveChat11() => ReceieveChat(11);
        public void ReceieveChat12() => ReceieveChat(12);
        public void ReceieveChat13() => ReceieveChat(13);
        public void ReceieveChat14() => ReceieveChat(14);
        public void ReceieveChat15() => ReceieveChat(15);
        public void ReceieveChat16() => ReceieveChat(16);
        public void ReceieveChat17() => ReceieveChat(17);
        public void ReceieveChat18() => ReceieveChat(18);
        public void ReceieveChat19() => ReceieveChat(19);
        public void ReceieveChat20() => ReceieveChat(20);
        public void ReceieveChat21() => ReceieveChat(21);
        public void ReceieveChat22() => ReceieveChat(22);
        public void ReceieveChat23() => ReceieveChat(23);
        public void ReceieveChat24() => ReceieveChat(24);
        public void ReceieveChat25() => ReceieveChat(25);
        public void ReceieveChat26() => ReceieveChat(26);
        public void ReceieveChat27() => ReceieveChat(27);
        public void ReceieveChat28() => ReceieveChat(28);
        public void ReceieveChat29() => ReceieveChat(29);
        public void ReceieveChat30() => ReceieveChat(30);
        public void ReceieveChat31() => ReceieveChat(31);
        public void ReceieveChat32() => ReceieveChat(32);
        public void ReceieveChat33() => ReceieveChat(33);
        public void ReceieveChat34() => ReceieveChat(34);
        public void ReceieveChat35() => ReceieveChat(35);
        public void ReceieveChat36() => ReceieveChat(36);
        public void ReceieveChat37() => ReceieveChat(37);
        public void ReceieveChat38() => ReceieveChat(38);
        public void ReceieveChat39() => ReceieveChat(39);
        public void ReceieveChat40() => ReceieveChat(40);
        public void ReceieveChat41() => ReceieveChat(41);
        public void ReceieveChat42() => ReceieveChat(42);
        public void ReceieveChat43() => ReceieveChat(43);
        public void ReceieveChat44() => ReceieveChat(44);
        public void ReceieveChat45() => ReceieveChat(45);
        public void ReceieveChat46() => ReceieveChat(46);
        public void ReceieveChat47() => ReceieveChat(47);
        public void ReceieveChat48() => ReceieveChat(48);
        public void ReceieveChat49() => ReceieveChat(49);
        public void ReceieveChat50() => ReceieveChat(50);
        public void ReceieveChat51() => ReceieveChat(51);
        public void ReceieveChat52() => ReceieveChat(52);
        public void ReceieveChat53() => ReceieveChat(53);
        public void ReceieveChat54() => ReceieveChat(54);
        public void ReceieveChat55() => ReceieveChat(55);
        public void ReceieveChat56() => ReceieveChat(56);
        public void ReceieveChat57() => ReceieveChat(57);
        public void ReceieveChat58() => ReceieveChat(58);
        public void ReceieveChat59() => ReceieveChat(59);
        public void ReceieveChat60() => ReceieveChat(60);
        public void ReceieveChat61() => ReceieveChat(61);
        public void ReceieveChat62() => ReceieveChat(62);
        public void ReceieveChat63() => ReceieveChat(63);
        public void ReceieveChat64() => ReceieveChat(64);
        public void ReceieveChat65() => ReceieveChat(65);
        public void ReceieveChat66() => ReceieveChat(66);
        public void ReceieveChat67() => ReceieveChat(67);
        public void ReceieveChat68() => ReceieveChat(68);
        public void ReceieveChat69() => ReceieveChat(69);
        public void ReceieveChat70() => ReceieveChat(70);
        public void ReceieveChat71() => ReceieveChat(71);
        public void ReceieveChat72() => ReceieveChat(72);
        public void ReceieveChat73() => ReceieveChat(73);
        public void ReceieveChat74() => ReceieveChat(74);
        public void ReceieveChat75() => ReceieveChat(75);
        public void ReceieveChat76() => ReceieveChat(76);
        public void ReceieveChat77() => ReceieveChat(77);
        public void ReceieveChat78() => ReceieveChat(78);
        public void ReceieveChat79() => ReceieveChat(79);
        public void ReceieveChat80() => ReceieveChat(80);
        #endregion
    }
}
```

근데 Data Dictionary를 쓰면서  
