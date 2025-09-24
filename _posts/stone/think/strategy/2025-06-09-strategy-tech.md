---
title: "250609 전략 - 테크"
description: "테크를 배우자"
categories: [수필, 생각]
tags: [Strategy]
image: "/assets/img/background/backtop.gif"

date: 2025-06-09. 21:36 # Init: from shader, effect, ref-idea, gwan
# last_modified_at: 2025-06-13. 00:29 # +공부 링크
# last_modified_at: 2025-06-18. 23:27 # +직장
# last_modified_at: 2025-08-11. 00:15 # +사냥 from 전략
last_modified_at: 2025-09-24. 22:34 # E -키워드 to strategy-tech-career
---

## 머리말

---

테크를 배우자. 근데 이제 흥미 포커스.  

## 메인

---

몬스터 -> UnitObject  
Die -> ByeBye
Spawn -> HiHi  
폴더 구조 정리  

- POT
- Simple Web Service
- Build Report -> Editor Log에 출력됨
- Input.mouseScrollData.y
  - x --> Apple Magic Mouse, Horizontal Scroll
- RectTransformUtility.RectangleContains(ScreenPoint)
  - Overlay 아닌 캔버스는 카메라 등록 필요
- Selection.objects -> 다중선택
- TextureImporter.getPlatformTextureSetting
- BackNavigationService -> 플랫폼/입력 상관없이 Back 개념 추상화
- Builder -> Popup / 객체(개념) -> View
- Adapter
  - /# if 지원기기
  - ~
  - /# elif
  - AdapterNone <-- 이것이 좋다
  - /# endif
- 파일 이름에 . 넣기?
- 프로퍼티 -> 상태 (개념적으로)
- 변수 -> 임시변수 뭐 이런 것들 있을 수 있으니까? 그런건가?
- 웹/서버
  - 네트워크 통신
  - admin

- Addressable
- CS
- 포폴 뭐로?
  - 수집형 하고 싶으면 수집형을
  - 서버
- 압축 포맷
  - ASTC 4x4
  - ETC2 8Bits
- Sprite/Default 차이
  - Atlas 압축 방시 ㄱ따라감?
- 잘못 커밋했을때
  - git log --oneline
  - chectkout <이동할브랜치>
  - cherrypick <커밋 hash>
- i.posworld.x,
- RotateCoords
  - float rad = radians (angle)
  - cosA = cos(rad)
  - sinA = sin(rad)
  - return float2(coords.x * cosA- coords.y * sinA, Coordsx * sinA + coords.y * cosA)
- URL 스킴
- mailto:
- Application.OpenURL

## 아트

---

- 아트 감성을 배워야 -> 단순 암기로 되는게 아님, 꾸준히 몇 년씩 그림그리고 해야

## 툴

---

### 유니티

#### 렌더 파이프라인

#### 6

### 포토샵

### 스파인

## TA

---

- [아르카도: 유니티로 배우는 3D 게임 이펙트 디자인](https://class101.net/ko/products/5f4b07c90c9c31008c937785)
- [Learn to Write Unity Compute Shaders](https://www.udemy.com/course/compute-shaders/?couponCode=PMNVD2525)
- [Learn Unity Shaders from Scratch](https://www.udemy.com/course/learn-unity-shaders-from-scratch/?couponCode=PMNVD2525)
- [Shader Development from Scratch with Cg - Unity 6 Compatible](https://www.udemy.com/course/unity-shaders/?couponCode=PMNVD2525)
- [고라니 셰이더 노드](https://www.youtube.com/watch?v=KnueAgpUL3Y)
- [레트로 셰이더&렌더링 에센스](https://www.youtube.com/playlist?list=PLctzObGsrjfyWa2CaxGtxsLD-W5zYC2JJ)
- [Catlike Coding](https://catlikecoding.com/unity/tutorials/)
- [Visual Case Studies](https://www.youtube.com/playlist?list=PLJ4rOFLQFH4C0zPBu-fgFKMvrHadY6dhm)
- [Alan Zucconi](https://www.alanzucconi.com/category/shader/)
- [Shaders Laboratory](https://www.youtube.com/@shaderslaboratory8437/videos)
- [URP recipe: Compute shaders](https://learn.unity.com/tutorial/urp-recipe-compute-shaders#)

- <https://blog.nexon.com/post/2321256>
- Game Feel, Juice

### 큰 방향성

- 지금
  - 그래픽스 기초 이론
  - 유니티 셰이더 코드
  - 파티클 시스템
  - 포토샵
- Reflection, CubeMap, Reflection Probe, MatCap

이론, 별개로 만드는 것이 필요.  
레퍼런스, 벤치마킹 하나 정해서 쭉 따라 만들 것.  
모든 것은 WM을 위하여. WM  

#### 이론

- 그래픽스 API
- **그래픽스 기초 이론**
  - 렌더링 파이프라인
  - 컬러 - 숫자
  - 비트
  - 텍스처와 rgb채널
  - 상수-변수
  - 리니어-감마
  - 사칙연산-항등원
  - 컬러 -> 값 변형
  - UV
  - 밉맵
  - 벡터
  - 벡터 연산
  - 선형보간 lerp
  - 분기

#### VFX

VFX를 만들기 위한 과정.  

- 셰이더
  - 툴
    - **셰이더 코드**
    - 셰이더 그래프
- 파티클:
  - 유니티: Shuriken (파티클 시스템) > VFX 그래프
    - Shuriken (파티클 시스템) -> 유튜브: 아르카도
  - 언리얼: Cascade > Niagara
- 리소스
  - 텍스쳐
    - **포토샵**, 섭스
    - tga, psd
  - 모델링
    - 3D Max, 블렌더
  - 후디니
- 이펙트
  - 타이밍
  - 형태
  - 색상
  - ...

#### 작업

- 모작

## 셰이더

---

### 배움

- 셰이더 프로그래밍만 할거라면 굳이 유니티를 쓸 필요 없고 webgl 이나 shadertoy 사이트에서 공부하는 게 더 낫다? 네이티브 그래픽스 api 다루는 게 낫다?

#### 참고 자료

- 유니티 기본 lit 뜯어보기?
- 비리비리 중국 ASE (엠플리파이 셰이더)
- <https://gamefx.co.kr/bbs/board.php?bo_table=ik>
- 블로그
  - ['대마왕':](https://chulin28ho.tistory.com/)
  - ['김포프': 셰이더 입문 강좌](https://blog.popekim.com/ko/tags/shader-book/)
  - ['Gusdnd_01':](https://velog.io/@gusdnd_01_11/posts)
  - ['캐니':](https://blog.naver.com/canny708/221547308831)
  - ['patriciogonzalezvivo'](https://patriciogonzalezvivo.com/)
  - ['jenlowe'](https://www.jenlowe.net/)
- 유튜브
  - ['RETR0': 셰이더 & 렌더링 에센스](https://youtu.be/4iSJW7YGrjY)
  - 아르카도 -> 이펙트 공부 순서 <https://youtu.be/LQPBRsgsoJ0>
    - [파티클기본설명 개정판 Ver.2019.4.1f1](https://youtu.be/2De-Bp262eE)
    - [07 게임이펙트제작시주의사항](https://youtu.be/TzGxJoHSrYo)
    - [게임 VFX의 예술적 원리](https://cafe.naver.com/unrealfx/20231)
    - [언리얼 이펙터가 꼭 알아야 할 색감과 타이밍 미리보기](https://youtu.be/VyvOfTGh3MM)
    - [Textures for VFX](https://simonschreibt.notion.site/Textures-for-VFX-Database-2c72eccccfa84a0eae927d778ad746cc)
    - [EffectTextureMaker](http://mebiusbox.github.io/contents/EffectTextureMaker/)
    - [Pinterest](https://www.pinterest.co.kr/)
    - [중국 이펙트 사이트](https://www.magesbox.com/)
    - [기적의 셰이더 그래프 '어떤 노드든 이해시켜 드립니다'](https://youtu.be/KnueAgpUL3Y)
    - [ben cloward](https://www.youtube.com/@BenCloward)
- 강의
  - 린반, 에반 (언리언), 쿠파 (유니티)
  - Learn to Write Unity Compute Shaders 유데미
  - [클래스101 중급 셰이더편](https://101creator.page.link/XfMv)
  - 유니티 샘플
- 책
  - 유니티 셰이더 스타트업
  - 대마왕의 유니티 URP 셰이더 그래프 스타트업
  - [The Book of Shader](https://thebookofshaders.com/?lan=kr)
    - 깊은 그래픽스보다는, 원하는 셰이더를 어떻게 만들것인가
    - 셰이더 잘한다 -> 셰이더를 어떻게 원하는 모양으로 만들까
    - 수학적 공식을 써서 패턴을 만드느냐는 이미 많은 사람들이 정리해뒀기때문에, 그 패턴을 알아야함

### 셰이더: 키워드

- 라이팅 모델
  - 디퓨즈(램버트?), 프레넬, 스펙큘러(퐁), 스펙큘러(블린-퐁), 디퓨즈(오렌-네이어)
- 라이팅심화
  - 탄젠트 노멀 매핑, 큐브맵 리플렉션, 실시간 그림자, 에디셔널 라이트
- fake PBR
  - PBR 물리기반렌더링, fakePBR (디퓨즈), fakePBR(스펙큘러), GGX 스펙큘러
- NPR (Non-Photorealistic)
  - NPR, NPR-Floor, " step/comparison, " smoothStep, "ramp, " Matcap, " 그림자 커스텀, " 외곽선
- 렌더링 파이프라인
  - 포워드 파이프라인, 디퍼드 ", 컴퓨트 셰이더, 버퍼
- 그 외
  - Distortion, GLSL, random, 컴퓨트 셰이더 (한별님 응원봉?)

## 디자인

---

- [타루님의 첫번째 콘서트 Data;Overflow에서 포스터 디자인를 레퍼런스로 인터미션, 오프닝 모션그래픽과 Smiley, We go 스크린 비디오 제작했습니다!](https://x.com/StudioLCM/status/1870445733436141859)
- [2024_12_26 블렌더 4일차\n\n렌더 세팅의 중요성 + 커피 모델링 / UV 제작](https://x.com/StudioLCM/status/1871987307831820483)
- [2024 SOA](https://x.com/goatdraw/status/1874078062180262108)
- [이세돌 뽑기! isedol Gacha! illust - @donmin_h animation - @drawsloth #이세돌 #이세계아이돌 #spine2d](https://x.com/drawsloth/status/1873614490216047020)
- 포퐅 사이트 디자인
  - <https://hyeon.me/>
- [Ok, here's another example of the animatic. As you can see, it doesn't match exactly with the final version. In this case I had to adjust the camera after the first pass of the animation to have it react more naturally to the action. Animation by Maxime Leclerc and Florian Durand](https://x.com/gintszilbalodis/status/1880155842441540064)
  - 일단 메모
- [涙](https://x.com/HDo2XtDXw6Xfy67/status/1881044949829107801)
- [](https://x.com/undefined/status/1883457662119202971)
- [](https://x.com/undefined/status/1894047031989014710)
- [](https://x.com/undefined/status/1895812892295118915)
- [](https://x.com/undefined/status/1895827183282852330)
- [testing cool ui](https://x.com/arxlight_vrc/status/1896225041169469617)
- [](https://x.com/OldInternetFeel/status/1899508734533009583)
- [진짜 초창기부터 비핸스 이용했어서 여기도 많이 봐주면 좋겠음 핀터 같은 여타 사이트랑 분위기나 올라오는 작업물이 완전히 다르기도 하고 특히 국가별/툴 별로 지정해서 볼 수 있는 게 짱조음 저는 그렇게 대만의 아티스트를 사랑하게 되었고....](https://x.com/huhcerealsly/status/1899374378057429371)
- [Week4【META=KNOT 2024 in AKASAKA BLITZ】](https://www.youtube.com/live/r4xvxXm7x9I?si=J33rcPIKXwjSRfGD)

## 그림

---

- [그림정보 아카이브](https://drawinggalleryarchive.tistory.com/)
- [클튜 - 그림 그리는 요령](https://tips.clip-studio.com/ko-kr)
- [클튜 - 그림 꿀팁사전](https://www.clipstudio.net/drawing)
- [픽시브 - 그림법](https://www.pixiv.net/howto)

- [피규어](https://www.amiami.com/eng/c/bishoujo/)
- [피규어](http://phatcompany.jp/)
- [피규어](https://fig-memo.com/)
- [피규어](https://www.hpoi.net/pic360/list)
- [포즈 생성](https://pose-trainer.com/)
- [퀵 포즈](https://pose-trainer.com/)
- [크로키 사이트 모음](https://blog.naver.com/salvia0623/220988400695)
- [인체공부 02.인체요약편](https://gall.dcinside.com/mgallery/board/view/?id=drawing&no=16717)
- [인체공부 03.포즈창작편](https://gall.dcinside.com/mgallery/board/view/?id=drawing&no=18001)
- [미술 해부학의 현대적 공부법](https://gall.dcinside.com/mgallery/board/view/?id=drawing&no=192865&page=3)
- [투시 공부법](https://gall.dcinside.com/mgallery/board/view/?id=drawing&no=193253&page=1)
- [투시 정리](https://gall.dcinside.com/mgallery/board/view/?id=drawing&no=199810&page=3)
- [투시](https://gall.dcinside.com/mgallery/board/view/?id=drawing&no=199800&page=3)
- [고질라군님 강좌 모음](https://bbs.ruliweb.com/hobby/board/300066/read/13203865?cate=322)
- [인체 비례 및 근육 간단 강좌](https://bbs.ruliweb.com/hobby/board/300066/read/1943462)
- [근육 움직임](http://kitasite.net/b/musmob/arm2/)
- [루미스](http://www.alexhays.com/loomis/)
- [초보자 채색, 명암 넣기](https://gall.dcinside.com/mgallery/board/view/?id=drawing&no=195382&page=1)

- [도트](https://gall.dcinside.com/mgallery/board/view/?id=pixelart&no=24728&page=1)
- [PixelArt Tutorials](https://www.slynyrd.com/pixelblog-catalogue)

- [그림 크로키](https://x.com/zm_zmfhzl/status/1823742377644974378)
- [야매 인체도형화](https://x.com/leyan91925680/status/1819006892888465785)

- 추상화 능력 (포인트), 자극 (평소 안그리던)
- 크로키
  - 한 선에 그리려고 노력
  - 전체적인 형태를 도형이나 실루엣으로 이미지화
  - 틀려도 지우지말고 옆에 이어서
  - 시간 다르면 그만큼 묘사도 다르게
  - 시간이 부족할 것 같으면 포인트에 집중
  - 시간이 부족할 것 같다면 얼굴 후순위로, 얼굴보다 손발 묘사가 도움된다
  - 그리는 대상은 3차원 세계의 입체
- Proko, Tutorial
- [How to Draw the Torso | Simplify Anatomy](https://youtu.be/qoZB9ieSVfs?si=iE1NP0AGDA3YCtq7)
- [채색 진짜진짜진짜진짜진짜진짜 쉬워짐 진짜진짜진짜진짜진짜진짜진짜진짜진짜진짜진짜진짜진짜진짜진짜진짜진짜진짜진짜진짜진짜진짜진짜진짜진짜진짜진짜진짜진짜진짜진짜진짜진짜진짜진짜진짜진짜진짜진짜진짜](https://youtu.be/5_WL8Nroav0?si=HNSEI5i58QqYXmlw)
- [귀 위치에 따른 얼굴 뒷모습](https://x.com/animesijyuku/status/1867857062903853192)
- [피규어 조각으로 인체 도형 익히기](https://x.com/gVMOUmbC5rdeJ05/status/1874303369541812707)
- [반대로 하면 됨](https://x.com/Rulin168/status/1875794156695798042)

## 애니메이션

---

## 에셋

---

### 영상

#### 이미지

- [Unsplash](https://unsplash.com/)
- [Pixabay](https://pixabay.com/)
- [Ezgif](https://ezgif.com/)

#### 아이콘

- [SimpleIcons](https://simpleicons.org/)
- [IcoonMono](https://icooon-mono.com/)
- [Flaticon](https://www.flaticon.com/)
- [Game-icons.net](https://game-icons.net/)

#### 에셋: 텍스쳐

- [AmbientCG](https://ambientcg.com/)
- [CGTrader](https://www.cgtrader.com/)
- [Textures](https://www.textures.com/)
- [3DSky](https://3dsky.org/)

#### 이미지 툴

- [Waifu2x - 이미지 업스케일링](http://waifu2x.udp.jp/index.ko.html)
- [Squoosh - 이미지 압축](https://squoosh.app/)
- [Materialize](https://boundingboxsoftware.com/materialize/)
- [NormalMap Online](https://cpetry.github.io/NormalMap-Online/)
- [Line2NormalMap](https://mttl9rtv.fanbox.cc/posts/7715867?utm_campaign=manage_post_page&utm_medium=share&utm_source=twitter)
- [Flamel(플라멜) - AI 이미지](https://flamel.app/)

#### 에셋: VFX

- [VFX Resource - For UnrealEngine](https://nielsdewitte.be/index.php?page=Pages/VFExtra.php)
- [EffectTextureMaker](https://mebiusbox.github.io/contents/EffectTextureMaker/)

### 소리

- [SoundEffect-Lab](https://soundeffect-lab.info/)

- [Bensound](https://www.bensound.com/royalty-free-music/track/memories)
- [99Sounds](https://99sounds.org/)
- [SoundBible](https://soundbible.com/)
- [FreeSound](https://freesound.org/)
- [FreeSFX](https://www.freesfx.co.uk/)
- [SoundJay](https://www.soundjay.com/)
- [Zapsplat](https://www.zapsplat.com/)
- [Dova-s](https://dova-s.jp/)
- [Soniss](https://sonniss.com/)
- [Soniss - gameaudiogdc](http://sonniss.com/gameaudiogdc#1605031061361-34588c70-73f2)
- [PlayOnLoop](https://www.playonloop.com/)
- [Soundsnap](https://www.soundsnap.com/)
- [FindSounds](https://www.findsounds.com/types.html)

- [사운드 리소스 모음](https://docs.google.com/spreadsheets/d/1GtehmgtnAX2dt5xM8Qv4Kj8-eZtGA5sRuCjw40oLI3o/edit#gid=0)
- [The Sound of MapleStory](https://larry.sh/post/the-sounds-of-maplestory/)
- [8-bit / 16-bit Sound Effects (x25) Pack](https://www.jdwasabi.com/store/8-bit-16-bit-sound-effects-x25-pack)
- sfxr: 8bit SFX 만드는 프로그램

- ["How Do I Get Good At Music?" Read: https://t.co/jEkjLqFZKl](https://x.com/tobyfox/status/754721147262992384)

### 3D 모델, 애니메이션

- [Sketchfab](https://sketchfab.com/)
- [Kitbash3D](https://kitbash3d.com/)
- [3DWareHouse](https://3dwarehouse.sketchup.com/)

- [Mixamo](https://www.mixamo.com/#/)

- [Dimensions - 휴먼덴시티](https://www.dimensions.com/)
- [3D API](http://3dapi.com/)

### 복합적

- [Kenney](https://kenney.nl/)
- [PolyHaven (텍스쳐, 3D)](https://polyhaven.com/)
- [SoundImage](https://soundimage.org/)
- [OpenGameArt](https://opengameart.org/)

### 웹

- [RealFaviconGenerator](https://realfavicongenerator.net/)
- [CSSGradient](https://cssgradient.io/)

### 툴, 레퍼런스

- [Internet Archive](https://archive.org/)

- [123APPS](https://123apps.com/ko/)
- [Convertio](https://convertio.co/kr/)
- ['diffchecker': 긴 문자열 비교 like linux diff](https://www.diffchecker.com/text-compare/)

- [Easings - 커브](https://easings.net/)
- [Desmos - 그래프 계산기](https://www.desmos.com/calculator?lang=ko)
- [모션 테이블](http://foxcodex.html.xdomain.jp/)

- [GAME UI DATABASE](https://www.gameuidatabase.com/)
- [Interface in Game](https://interfaceingame.com/)
- [게임 UI-UX 자료 모음](https://boom-seeder-9ee.notion.site/UIUX-dcfa267e96aa4679b0c0622a99d3ceaa)
- [영화 스냅샷, 컬러](https://screenmusings.org/)
- [게임 레벨 디자인](https://noclip.website/)

### Asset 메모

- <https://jdsherbert.itch.io/minigame-music-pack>
- <https://shapeforms.itch.io/shapeforms-audio-free-sfx>
- <https://crazy-potato-game-studio.itch.io/medieval-fantasy-16-x-16-pixel-art-items>
- <https://bdragon1727.itch.io/pixel-holy-spell-effect-32x32-pack-3>
- <https://sami-hiltunen.itch.io/free-audio-asset-collection>
- <https://kronbits.itch.io/pixatool>
- <https://kronbits.itch.io/>
- <https://screamingbrainstudios.itch.io/>
- <https://grafxkid.itch.io/>
- <https://thkaspar.itch.io/micro-character-bases>
- <https://thkaspar.itch.io/tth-animals>
- <https://rhosgfx.itch.io/>
- <https://luizmelo.itch.io/>
- <https://snoopethduckduck.itch.io/>
- <https://butterymilk.itch.io/>
- <https://codemanu.itch.io/>
- <https://henrysoftware.itch.io/>
- <https://wenrexa.itch.io/>
- <https://kenney.nl/>
- <https://axulart.itch.io/>
- <https://randallcurtis.itch.io/16-bit-rpg-icons>
- <https://caz-creates-games.itch.io/>
- <https://pebonius.itch.io/surtizens>
- <https://egordorichev.itch.io/chare>
- <https://blacis.itch.io/pixel-monsters-mega-pack>
- Kenny
  - <https://kenney.nl/assets/ui-pack-rpg-expansion>
- [텍스트 생성기](https://perchance.org/useful-generators)
- 16x16 Assorted RPG Icons, 16x16 Weapons RPG Icons

## 메모

---

### 레퍼런스

- 롤 effect?
- 모두의 마블 주사위
- [Effect](https://x.com/MrB_Jensen/status/1792479223866589670)
- [Effect](https://x.com/GabrielAguiarFX/status/1781339488679075911)
- [Effect](https://x.com/cmzw_/status/1834555458444763276)
- [시간 이펙트](https://x.com/cmzw_/status/1793318381313278205)
- [웜홀?](https://x.com/Indiedev_Hub/status/1792867212857950564)

### 도토리

- [블렌더 에드온](https://x.com/h_ram01/status/1545646179488124928?s=20&t=T5ZiW47P8k2CtzxR9_ec_g)
- [애니메이션의 타이밍](https://spine304.tistory.com/65)
- [AO 러프니스 메탈릭 채널이 정해진 이유](https://youtu.be/ZQagb9WG1bg?si=weV2vFkrYZcdyvTd)
- [그래픽 블로그](https://rusalgames.tistory.com/)
- [투명한 메쉬로 그림자 표현하기](https://x.com/JasperRLZ/status/1182510103943094272?ref_src=tws5Etfwrc%)
- [Z-Fighting 고치는 방법](https://x.com/FreyaHolmer/status/799602767081848832?s=20&t=EBmnPU-IlwzD5ylVXmrPqQ)
- [블렌더 에드온 오브젝트 다른 오브젝트에 달라붙게](https://x.com/BlenderHub7/status/1819801883109695547)
- [블렌더 루프 선택에서 특정 페이스만 띄워서 선택하기](https://x.com/what_wat_/status/1818966855127838897)
- [모션그래픽](https://x.com/MotionLCM/status/1820878279165014349)
- [빛](https://x.com/imo_dekai/status/1804506139809431972)
- [ai로 만든 로고들](https://x.com/VadimCarazan/status/1778371364514078766)
- [Blender - Rigidbody Collision](https://youtu.be/UI_Fntqj8Eo?si=tjyaL3x88dj_prjj)
- [Blender - Particle System](https://youtu.be/lMNxWMhjMpU?si=m-VfRIsUsbBmfhsi)
- Ripple 파문
- UIOutline 에셋에 텍스쳐 이상한거 넣으면 멋잇다
- BaseMeshEffect
- Graphics
- VertexHelper
- Vertex 수정. 위치,색.
- Unity Vertex Animation Texture ?
- VFX GPU, Particle CPU
- UI Effect Github
- 프로테제 효과

## 사냥

---

할 것 만들기.  

- **Blog**
  - review
  - mindset -> strategy 병합 필요
  - 글 내용 정리 (Project, 일반): (수가 많아서, 천천히 부지런히)
  - 글 읽을 수 있게 만들기
    - 메모 형식에 가까운 것들을 시작과 끝이 있는 글로 만들기
    - 마치 발표자료 처럼
    - 메모 형식이 맞는 목적의 글도 있지만, 그렇지 않은 글도 메모 형식인것이 문제
  - [**reference-idea**](/posts/reference-idea): [](/_posts\witch-mendokusai\world\2023-01-27-reference-idea.md)
  - `gwan.md`
- **WM**:
  - item-object shine-shader
    - 기존 billboard shader-code -> shader-graph converting
      - [가져온 code 이해](https://darkcatgame.tistory.com/137)
        - 행렬 공부
          - 회전 행렬
  - 시간
    - 어떻게 계산하는지
    - 시간, 시계, 동주기자전, 위성 (태양과 달을 대신 할)
    - 계절 (동주기자전 기준)
  - 시작
    - WM의 시점
    - 하루: 위성을 통해, 달/해 -> 어떻게?
    - 국제 표준, 활발히 활동/교류?
      - 언어? 마법으로 시각적 표현? 콘택트?
  - 마을
    - Pokke Village
- [**Woodon**](/posts/woodon): [](/_posts/works/virtual/woodon/2024-10-25-woodon.md)
- 정리
  - 구글 드라이브 정리
  - 트위터 팔로우 정리 (레퍼런스)
  - 그림 북마크 정리
- 세상
- 공부
  - UI Toolkit
  - Midi, 음악
  - Font, OTF TTF
  - Unity Web: x Github Page
    - Like 전시회/미술전 ?
  - PS
- 책
- 그림
- community
  - 강연
    - unite, gdc 등
  - unity
    - unity square
    - unity document
    - unity roadmap
    - unity how.to
    - learning resources: unity 6 graphics
    - 유니티 설계 경험 기록
  - github
    - star
    - `개발자`, `프로그래머`, `면접`, `컴공` 같은 키워드로 repository 검색
  - follow
    - [GamePix](https://x.com/G_P_Art): 250527
    - [Wonpuri](https://x.com/Wonpuri): 240412
  - site
    - [Coloso](https://coloso.co.kr/)
    - [텀블벅](https://tumblbug.com/discover?tab=category&category=video-games)
    - [위키독스 - 온라인 책 공유 플랫폼](https://wikidocs.net/)
    - [교보문고](https://ebook.kyobobook.co.kr/dig/pnd/showcase?pageNo=3819&cmdt=EBK&clst1=21&clst2=&clst3=&landing=Y)
    - velog: [suhan0304](https://velog.io/@suhan0304/posts)
    - midium: 트라플라
    - microsoft:
      - CPU 인사이트: Enum ToString이 Reflection을 쓴다?
      - visual studio document - cpu insight, 성능 insight, enum.toString()~
  - discord
    - VAULT
    - 어레이의 개발문고 크루
    - Official Unity Discord
  - blog
    - 전문가가 쓴 글 읽기
      - 이론을 이해하는 것과 관용적인 쓰임새를 이해하는 건 다른 문제
      - 혼자서는 얻기 어려운 깊은 통찰
      - 기술에 대해 전문가가 어떤 문제나 주장을 제기하는지 알기 -> 이해 깊어짐
    - 개발사/개발자 블로그
      - nexon, cookApps
    - [슈퍼코믹](https://blog.naver.com/ekfvoddl3535)
    - [메이플스토리 블로그](https://blog.maplestory.nexon.com/)
    - [BatStudio](https://www.ibatstudio.com/)
    - [풍풍풍(sorkelf)](https://blog.naver.com/sorkelf)
    - [정대찬 - 정대찬의 개발 일지](https://24dc-m.tistory.com/)
    - [망나니 개발자](https://mangkyu.tistory.com/category)
    - [BBAGWANG](https://bbagwang.com/posts/)
    - [잇창명](https://eatchangmyeong.github.io/)
    - [이고드](https://dogy3045.tistory.com/): 게임 아티클
    - [K리그 프로그래머](https://jeho.page/)
    - [한별 - 한별이의 메모장](https://blog.naver.com/twinkle_onestar): 셰이더
    - [원소랑 - 게임 만드는 원소랑](https://blog.naver.com/sorang226/221709362869)
    - [펩시맨(izure) - 경어와 반말이 오가는 블로그](https://blog.naver.com/izure)
    - [대그 - Daeg Game Studio](https://blog.naver.com/mbjjang0321)
    - [eeeuns](https://eeeuns.github.io/)
    - [Nauts - Nauts의 게임음악 이야기](https://blog.naver.com/supernauts)
    - [Ju Hwijung - 개발창고](https://blog.juhwijung.com/)
      - [Ju Hwijung - 플밍일기](https://blog.naver.com/5755084)
    - [대마왕 - 대충 살아가는 게임개발자](https://chulin28ho.tistory.com/)
    - [Arizen - Local](https://blog.naver.com/dkflwps/223623274650)
      - [Arizen - Local](https://w0lf.kr/pages/index)
    - [ANDMoonY 앤무니](https://blog.naver.com/PostList.naver?blogId=myoh8901)
    - [산적대왕](https://blog.naver.com/raveneer)
    - [댄싱돌핀](https://blog.naver.com/jysa000)
    - [Wookje](https://wookje.dance/)

### 블로그 글

- `self-qna.md` 정리: Career 병합한 부분을 주제에 맞게 다듬기.
- `unite.md` 정리: Unity에 일부 세션 영상 올라옴. 참고할 것.
- \@GWAN
- \#Q

### 역사

```yml
Effect
date: 2024-11-13. 06:50 # Init
last_modified_at: 2025-05-28. 06:15 # +메모 from career-learning
last_modified_at: 2025-06-08. 20:18 # +메모
```

2025-06-09. 22:21: 글 확장,  
`2024-11-13-effect-DRAFT: Effect`.

### 기록
