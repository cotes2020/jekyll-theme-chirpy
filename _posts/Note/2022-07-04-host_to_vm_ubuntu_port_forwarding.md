---
title: Windows에서 hyper-v ubuntu로 포트포워딩 (netsh)
date: 2022-07-04-13:54  +0900
categories: [Note,hyper-v]
tags: [netsh, port forwarding]
---

## Host에서 가상머신으로 포트포워딩
<hr style="border-top: 1px solid;"><br>

+ Step 1. 공유기 내에서 내 PC로 포트포워딩 설정
  + CMD에서 ipconfig 명령어를 통해 내 PC의 게이트웨이 주소를 확인한다.
  
  + 확인한 후 공유기에 접속하여 나의 경우엔 LG 공유기라서 공유기 설정을 변경하면 공유기를 재시작하기 때문에 DHCP 기능으로 IP가 변경이 되므로 DHCP 할당 정보에서 내 PC의 IP를 고정시켜준다.
  
  + 고정 시킨 후 네트워크 설정에서 NAT 기능을 설정해준다. 외부에서 들어올 포트 범위를 입력하고 내 PC IP를 입력해준 뒤 내부 포트를 지정해주면 된다.

<br>

+ Step 2. netsh로 내 PC에서 가상머신으로 포트포워딩

  + ```netsh interface portproxy listenport={외부에서 들어올 port} listenaddress={내 PC IP} connectport={가상머신에서 연결할 port} connectaddress={가상머신 IP}```
  + 나의 경우엔, 가상머신의 DHCP 기능을 없애지 않아서 변경될 수 있는데 아직까진 PC를 껐다 켜도 변경되지 않았지만 변경되면 다시 설정해주면 된다.

<br>

+ Step 3. Hyper-v Ubuntu에서 확인
  + ```nc -lvp {connectport}```을 해준 뒤 내 PC에서 ```내 IP:listenport```로 연결해준 뒤 포트포워딩이 됬는지 확인해준다. 

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 참고
<hr style="border-top: 1px solid;"><br>

Link
: <a href="https://psj-study.tistory.com/243" target="_blank">psj-study.tistory.com/243</a>
: <a href="https://kaka09.tistory.com/50" target="_blank">kaka09.tistory.com/50</a>
: <a href="https://hsunnystory.tistory.com/89" target="_blank">hsunnystory.tistory.com/89</a>

<br><br>
<hr style="border: 2px solid;">
<br><br>
