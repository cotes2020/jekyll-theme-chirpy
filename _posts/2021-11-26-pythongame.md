### 파이썬을 이용해서 간단하고 쉬운 게임 만들어보기

## 작품 소개
: 1학년 2학기, 공학도를 위한 창의적 컴퓨팅이라는 강의를 들으면서 파이썬이라는 컴퓨팅 언어를 배웠는데, 이를 이용해서 각종 복잡한 활동들을 할 수 있으며 게임도 만들어낼 수 있다는 교수님의 말씀을 들었다. 예전부터 게임에 관심도 많고 좋아하기도 했으며, 실제 파이썬을 통해 게임을 만들었을 때, 내가 배운 내용이 실제로 사용되는지 궁금해서 한 번 파이썬을 이용해서 게임을 만들어보았다.
 이 코드는 유튜브 “초보 코딩”이라는 채널, 이수안 컴퓨터 연구소에서 배포한 pyshooting이라는 pdf 파일, 저서 “혼자 공부하는 파이썬”, 강의 시간에 배운 내용으로 Anaconda3의 jupiter notbook을 통해 제작했다. 그리고 과거에 내가 했던 드래곤 플라이트라는 게임을 모티브로 삼아서 이를 비슷하게 만들어보고 싶었다.

## 기존 목표
: 플레이어가 조종하는 드래곤 형태의 캐릭터를 화살표 상, 하, 좌, 우를 이용해서 움직이고 스페이스 바를 눌러서 불을 뿜게 마시며 위에서부터 아래로 적들이 출몰하고 적과 충돌하면 게임 오버되게 만드는 것이 기본적인 게임의 구조이다. 거기다가 시간, 점수, 놓친 적의 숫자, 좌우에서부터 발사되는 장애물, 획득 시 몇 가지 좋은 효과를 내는 소모성 효과를 지닌 것과 나쁜 효과를 내는 소모성 효과를 
지닌 것이 드문 확률로 맵 어딘가에 등장하게 만들어보는 것 

## 결과
: 코드 
```
import pygame
import random
import time
from datetime import datetime

# 1. 게임 초기화
pygame.init()

# 2. 게임창 옵션 설정
size = [400, 800]
screen = pygame.display.set_mode(size)

title = "Dragon Flight"
pygame.display.set_caption(title)

# 3. 게임 내 필요한 설정
clock = pygame.time.Clock()

class obj:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.mobe = 0
    def put_img(self, address):
        if address[-3:] == "png":
            self.img = pygame.image.load(address).convert_alpha()
        else :
            self.img = pygame.image.load(address)
        self.sx, self.sy = self.img.get_size()
    def change_size(self, sx, sy):
        self.img = pygame.transform.scale(self.img, (sx, sy))
        self.sx, self.sy = self.img.get_size()
    def show(self):
        screen.blit(self.img, (self.x,self.y))

# a.x-b.sx <= b.x <= a.x+a.sx
# a.y-b.sy <= b.y <= a.y+a.sy

def crash(a, b):
    if (a.x-b.sx <= b.x) and (b.x <= a.x+a.sx):
        if (a.y-b.sy <= b.y) and (b.y <= a.y+a.sy):
            return True
        else:
            return False
    else:
        return False
        
ss = obj()
ss.put_img("C:/Python39/images/ss.png")
ss.change_size(60,80)
ss.x = round(size[0]/2- ss.sx/2)
ss.y = size[1] -ss.sy -100
ss.move = 6     

left_go = False
right_go = False
up_go = False
down_go = False
space_go = False

m_list = []
a_list = []

black = (0,0,0)
white = (255,255,255)
k = 0

kill = 0
loss = 0

#4-0. 게임 시작 대기 화면
SB = 0
while SB == 0:
    clock.tick(60)
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                SB = 1
    screen.fill(black)
    font = pygame.font.Font("C:/Python39/Fonts/나눔 글꼴/나눔고딕/NanumGothic.otf", 15)
    text = font.render("PRESS SPACE KEY TO START", True, (255,255,255))
    screen.blit(text, (40, size[1]/2-50))
    pygame.display.flip()
        
    

# 4. 메인 이벤트
start_time = datetime.now()
SB = 0
while SB == 0:

    # 4-1. FPS 설정
    clock.tick(60)

    # 4-2. 각종 입력 감지
    # ss: 인물 키보드 입력 상하좌우 이동
    # ss1: 던져지는 교재 마우스 클릭, 마우스 위치
    # ss2: 과제 빌런 랜덤 발생 랜덤 일자 이동
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            SB = 1
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                left_go = True
            elif event.key == pygame.K_RIGHT:
                right_go = True
            elif event.key == pygame.K_UP:
                up_go = True
            elif event.key == pygame.K_DOWN:
                down_go = True
            elif event.key == pygame.K_SPACE:
                space_go = True   
                k = 0
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT:
                left_go = False
            elif event.key == pygame.K_RIGHT:
                right_go = False
            elif event.key == pygame.K_UP:
                up_go = False
            elif event.key == pygame.K_DOWN:
                down_go = False
            elif event.key == pygame.K_SPACE:
                space_go = False    
                

    # 4-3. 입력, 시간에 따른 변화
    now_time = datetime.now()
    delta_time = (now_time - start_time).total_seconds()
    
    if left_go == True:
        ss.x -= ss.move
        if ss.x <= 0:
            ss.x = 0
    elif right_go == True:
        ss.x += ss.move
        if ss.x >= size[0] - ss.sx:
            ss.x = size[0] - ss.sx
    elif up_go == True:
        ss.y -= ss.move
        if ss.y <= 0:
            ss.y = 0
    elif down_go == True:
        ss.y += ss.move
        if ss.y >= size[0] + 3*ss.sy -10:
            ss.y = size[0] + 3*ss.sy -10
               
    if space_go == True and k % 6 == 0:
        mm = obj()
        mm.put_img("C:/Python39/images/mm.png")
        mm.change_size(30,30)
        mm.x = round(ss.x + ss.sx/2 - mm.sx/2)
        mm.y = ss.y - mm.sy
        mm.move = 15
        m_list.append(mm)
    k += 1    
    d_list = []
    for i in range(len(m_list)):
        m = m_list[i]
        m.y-= m.move
        if m.y <= -m.sy:
            d_list.append(i)
    for d in d_list:
        del m_list[d]

    if random.random() > 0.98 : 
        aa = obj()
        aa.put_img("C:/Python39/images/aa.png")
        aa.change_size(60,80)
        aa.x = random.randrange(0, size[0]-aa.sx-round(ss.sx/2))
        aa.y = 10
        aa.move = 3
        a_list.append(aa)
    d_list = []
    for i in range(len(a_list)):
        a = a_list[i]
        a.y += a.move
        if a.y >= size[1]:
            d_list.append(i)
    for d in d_list:
        del a_list[d]
        loss += 1
            
    dm_list = []
    da_list = []
    for i in range(len(m_list)):
        for j in range(len(a_list)):
            m = m_list[i]
            a = a_list[j]
            if crash(m,a) == True:
                dm_list.append(i)
                da_list.append(j)
    dm_list = list(set(dm_list))
    da_list = list(set(da_list))

    for dm in dm_list:
        del m_list[dm]
    for da in da_list:
        del a_list[da]
        kill += 1
        
    for i in range(len(a_list)):
        a = a_list[i]
        if crash(a, ss) == True:
            SB = 1
            GO = 1
    
    # 4-4. 그리기
    screen.fill(black)
    ss.show()
    for m in m_list:
        m.show()
    for a in a_list:
        a.show()
        
    font = pygame.font.Font("C:/Python39/Fonts/나눔 글꼴/나눔고딕/NanumGothic.otf", 20)
    text_kill = font.render("killed: {}, loss: {}".format(kill, loss), True, (0,255,0))
    screen.blit(text_kill, (10, 5))
    
    text_time = font.render("time : {}".format(delta_time), True, (255,255,255))
    screen.blit(text_time, (size[0]-100, 5))
    
    # 4-5. 업데이트
    pygame.display.flip()

# 5. 게임 종료
while GO == 1:
    clock.tick(60)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            GO = 0
    font = pygame.font.Font("C:/Python39/Fonts/나눔 글꼴/나눔고딕/NanumGothic.otf", 15)
    text = font.render("GAME OVER", True, (255,0,0))
    screen.blit(text, (80, size[1]/2-50))
    pygame.display.flip()
pygame.quit() 
```

이미지:

  ![묶음 개체입니다.](file:///C:\Users\user\AppData\Local\Temp\DRW000007e8230f.gif)  



## 개선해야 할 점

: 기본적인 구조는 내가 처음에 구상한 것처럼 만들 수 있었고, 시간, 점수, 놓친 횟수 같은 것은 구현했지만 내가 창의적으로 만들어냈다기 보다는 유튜브, 책 등의 도움을 많이 받으며 이를 거의 따라 만든 것과 다를 바가 없다. 캐릭터를 움직이게 하거나 게임을 시작하고 종료할 때 등장하는 텍스트 등, 그리고 일부 내용들은 내가 배운 내용을 응용해서 제작해보았다.
 또한 내가 처음에 추가하고자 했던 랜덤성 아이템의 등장과 추가적인 장애물의 기능은 추가하지 못했으며 플레이어 캐릭터가 발사하는 불덩이 파일에서 하얀 부분을 없애는 방법을 적용하지 못해서 발사체가 조금 이상하다.
 그리고 텍스트가 화면의 중간보다 살짝 치우쳐진 곳에 출력되게 되었고, 적들에게 플레이어가 충돌하는 범위가 캐릭터의 크기보다 살짝 더 커서 충돌 판정이 굉장히 애매하고 확실하지 않았다.