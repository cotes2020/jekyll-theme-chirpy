---
layout: post
title: 音乐偏好分析
date: 2021-08-31 14:36:35 +0800
Author: Sokranotes
tags: [recording]
comments: true
categories: recording
typora-root-url: ..
---

## 摘要

为了

- 根据自己听过的作品进行了解自己的音乐偏好；
- 以便未来更加有条理，更加全面地同朋友们分享古典音乐；

爬取到了自己的网易云音乐歌单数据，本文将对歌单数据进行数据分析。

## 引言

《杜伊诺哀歌》中有一句话[^1]：“机灵的禽兽注意到，我们并不以这个被解释的世界为家”。“人是可以解释这个世界，但是人就能以此为家吗？”[^2]我也曾思索过人生整体的意义[^3]，最终只能得出结论：整个的人生是没有意义的，但是生命中的每件事都是有意义的。虽然整个的人生是没有意义的，可是我们依旧发自本能而且坚定不移的，迎着其巨大的冲击往前走。而情感，艺术，意义“提供了这个理解可以被悬置的空间”。[^2]

古典音乐[^4]也给我提供了这样一个空间。在接触古典音乐之前，从都不觉得有什么东西能够陪伴人一生。享受古典音乐带来的超越时间和空间的感动，慰藉，欢喜，鼓舞，共鸣[^5]之后，我确立了一个信念：无论发生什么事，音乐不会抛弃你，音乐会一直陪着你。

最近（2021年）发现自己更喜欢Spotify的用户体验，便决定更换流媒体音乐软件，但是所有的歌单数据全在网易云音乐上面，使用Spotify又要重新建立自己的歌单。总结自己的古典音乐就成了很自然的事情。

而且好的东西应该分享。科普兰说：“如果你想更好的了解音乐，除了听，别无他途“[^6]。可是以往自己分享的曲目都零零散散，一些我当下非常喜欢的曲目，对于不了解的朋友并不是那么通俗，不太容易被接受。并不是不愿意分享，所以我想分享也需要讲策略。

为了根据自己听过的作品进行了解自己的古典音乐偏好，同时更加有条理，更加全面地同朋友们分享古典音乐，我从网易云音乐上面爬取到了自己的歌单数据，并在本文中展开数据分析。

## 数据介绍

从2016年6月开始使用网易云音乐，截止到2021-07-28，5年间，我在网易云音乐累积听歌5216首。

从中导出的数据包括3个部分

1. 听歌排行中的100条记录；
2. 自己创建的歌单共389个，13497条记录；
3. 收藏的歌单64个，1200条记录。

本文将分别针对这几部分数据进行分析。

## 听歌排行TOP100

本段将针对听歌排行中的100条记录进行分析。

主要包括如下字段：播放次数playCount，曲名name，时间dt，艺术家名称ar names，专辑标识al id，版权信息copyright。

#### 播放次数分析

图1中，次数前5多的曲目标记为金黄色，前20中的其他曲目标记为蓝色，绿色标记为排名第50的曲目。TOP100所有曲目总播放次数达23519次，平均播放次数235.19次，前22首曲子的播放次数超过了平均，包含4首纯音乐。其中7首曲子名字过长被遮挡，见[遮挡曲目](#遮挡曲目)。

<img src="/assets/img/2021-07-28-my netease cloud playlist data analysis/图1次数图.png" alt="次数图" style="zoom: 25%;" />

<center>图1 次数图</center>

从图2中可看出，播放次数呈现长尾分布，其中78%的曲目落在[106, 216]这个区间当中，其中19%落在(216, 545]这个区间中，而只有3%落在区间(600, 2626]中。

![次数分布图](/assets/img/2021-07-28-my netease cloud playlist data analysis/图2次数分布图.png)

<center>图2 次数分布图</center>

如图3所示，在这100首曲子中，排名前5的曲目的播放次数占到了总播放次数的25%，前20占到了总数的48%，前50占到了总数的72%。

<img src="/assets/img/2021-07-28-my netease cloud playlist data analysis/图3次数占比图.png" alt="占比图" style="zoom: 80%;" />

<center>图3 次数占比图</center>

#### 播放时间分析

图4中的颜色表示同上。所有曲目总播放时间达1922.7小时，平均播放19.227小时。前28首曲子的播放时间超过了平均，包含2首纯音乐。

<img src="/assets/img/2021-07-28-my netease cloud playlist data analysis/图4总时间图.png" style="zoom:25%;" />

<center>图4 总时间图</center>

从图5中可看出，时间分布同播放次数分布同相同，也呈现长尾分布，但较次数分布相比更加缓和，其中72%的曲目落在(3, 20]这个区间当中，其中26%落在(20, 80]这个区间中，而只有2%落在区间(80, 210]中。

![时间分布图](/assets/img/2021-07-28-my netease cloud playlist data analysis/图5时间分布图.png)

<center>图5 时间分布图</center>

由图6可知，时间占比图同次数占比图类似，但是时间分配不平衡更加明显，100首曲子中，排名前5的曲目的播放次数占到了总播放次数的27%，前20占到了总数的55%，前50占到了总数的81%。

<img src="/assets/img/2021-07-28-my netease cloud playlist data analysis/图6时间占比图.png" style="zoom:80%;" />

<center>图6 时间占比图</center>

#### 音乐风格分析

100首曲子中包含古典音乐[^4]86首，纯音乐10首，流行歌曲4首。

从播放次数上看，由图7和图8可知，纯音乐的平均次数最高，但是纯音乐总次数占比只有7.9%，而古典音乐总次数占比为87.6%，且平均次数也较高，为237.7次。

![不同风格音乐次数图](/assets/img/2021-07-28-my netease cloud playlist data analysis/图7不同风格音乐次数图.png)

<center>图7 不同风格音乐次数图</center>

![不同风格音乐总次数占比图](/assets/img/2021-07-28-my netease cloud playlist data analysis/图8不同风格音乐总次数占比图.png)

<center>图8 不同风格音乐总次数占比图</center>

从时间上看，由图9和图10可知，古典音乐的平均时间为20小时，总时间达1731.2小时，占比90%。

综合次数和时间两方面可以得出结论，总体上音乐风格偏爱古典音乐，也包含部分纯音乐，流行音乐。

![不同风格音乐时间图](/assets/img/2021-07-28-my netease cloud playlist data analysis/图9不同风格音乐时间图.png)

<center>图9 不同风格音乐时间图</center>

![不同风格音乐总时间占比图](/assets/img/2021-07-28-my netease cloud playlist data analysis/图10不同风格音乐总时间占比图.png)

<center>图10 不同风格音乐总时间占比图</center>

## 古典音乐分析

在听歌排行TOP100中，古典音乐共[^4]86首，分别来自23位不同的作曲家。

#### 作曲家总览

如图11，用不用颜色对不同时期的作曲家加以区分：绿色表示巴洛克（Baroque）时期，金黄色表示古典主义（Classical）时期，粉红色表示浪漫主义（Romantic）时期，蓝色表示20世纪（20st century）。23位作曲家作品的时间跨度接近500年。

![作曲家时代总览图](/assets/img/2021-07-28-my netease cloud playlist data analysis/图11作曲家时代总览图.png)

<center>图11 作曲家时代总览图</center>

#### 播放次数分析

由图12可知，依据播放次数来看，Johann Sebastian Bach，Antonio Lucio Vivaldi以及Ludwig van Beethoven的作品听得最多，远超其他作曲家的作品，曲目分别有21首，18首以及11首。

![作曲家次数分布图](/assets/img/2021-07-28-my netease cloud playlist data analysis/图12作曲家次数分布图.png)

<center>图12 作曲家次数分布图</center>

而如图13，依据占比图可知，J.S.Bach的作品在86首古典作品中占24%，Vivaldi和Beethoven的作品分别占21%和13%。这三位作曲家的作品在全部古典作品中总占比58%，而作品出现次数更少的作曲家们的占比约为：4次9%，3次11%，2次9%，1次13%。

![作曲家次数占比图](/assets/img/2021-07-28-my netease cloud playlist data analysis/图13作曲家次数占比图.png)

<center>图13 作曲家次数占比图</center>

#### 播放时间分析

![作曲家时间分布图](/assets/img/2021-07-28-my netease cloud playlist data analysis/图14作曲家时间分布图.png)

<center>图14 作曲家时间分布图</center>

![作曲家时间占比图](/assets/img/2021-07-28-my netease cloud playlist data analysis/图15作曲家时间占比图.png)

<center>图15 作曲家时间占比图</center>

#### 音乐时期分析

根据图14可知，其中巴洛克时期听得作品次数最多，其次是浪漫主义时期，然后是古典主义时期，最后是20世纪作曲家的作品，其中97%分布在巴洛克时期，古典主义时期以及浪漫主义时期。

![作曲家次数占比图](/assets/img/2021-07-28-my netease cloud playlist data analysis/图16作曲家次数占比图.png)

<center>图16 作曲家次数占比图</center>

但是据图17可知，虽然浪漫主义时期作品次数只占27%，但是时间占比却达到了48%，而作品次数占比最高的巴洛克时期占比51%，但是总时间却只占比30%，说明浪漫主义时期上榜TOP100的作品虽然次数不如巴洛克时期作品，但是单曲时长比较长。

![作曲家时间占比图](/assets/img/2021-07-28-my netease cloud playlist data analysis/图17作曲家时间占比图.png)

<center>图17 作曲家时间占比图</center>

由图18也可以证明浪漫主义时期单曲时长较巴洛克时期相比更长，为6'32''，从这幅图中也能看出，音乐风格倾向于单曲时长较长的曲子，这其中也是因为通常一部作品由几个乐章组成，而单个乐章的长度同流行歌曲相比也非常长。不过其中也不乏一些短小的作品，如巴洛克时期维瓦尔第的四季，李斯特和肖邦的一些较短的钢琴练习曲等。

![作曲家作品平均时间图](/assets/img/2021-07-28-my netease cloud playlist data analysis/图18作曲家作品平均时间图.png)

<center>图18 作曲家作品平均时间图</center>

#### 按编制划分

根据曲目的编制对曲目进行手动划分，得到下图19，不难看出，更偏向于大编制的协奏曲和管弦曲目，solo也占据了一席之地，但是室内乐和solo这样的小编制还是略少与大编制的协奏曲和管弦曲目。

![不同编制曲目占比图](/assets/img/2021-07-28-my netease cloud playlist data analysis/图19不同编制曲目占比图.png)

<center>图19 不同编制曲目占比图</center>

## 后记

一开始是想对自己的音乐偏好做一个数据分析，并且分享古典音乐。数据分析的部分其实在2021年就做完了，音乐推荐部分一直没写，但是写了音乐推荐才发现，古典音乐曲库和作曲家太庞大了，这部分太难驾驭了，还是决定分开好了~

感谢你的阅读，望批评指正~

## 附录

### 遮挡曲目

J.S. Bach: Concerto in D Minor, BWV 974 - for Harpsichord/Arranged by Bach from: Oboe Concerto in D minor by Alessandro Marcello (1685-1750) - 2. Andante

Piano Concerto No. 1 in B-Flat Minor, Op. 23, TH. 55:1. Allegro non troppo e molto maestoso - Allegro con spirito

8 Humoresques, Op. 101, B. 187:No. 7, Poco lento e grazioso (Transcribed by Oscar Morawetz for Violin, Cello & Orchestra)

Cantata No.140: Wachet auf, ruft uns die Stimme (27th Sunday after Trinity), BWV140: i. Chorus: Wachet auf, ruft uns die Stimme

Variations On A Theme By Haydn "St. Anthony Variations" Op. 56b:Chorale St. Antoni: Andante (Live At Grosses Festspielhaus, Salzburg / 2009)

Symphony No. 6 in F Major, Op. 68, Pastoral:I. Allegro ma non troppo (Erwachen heiterer Emfindungen bei der Ankunft auf dem Lande)

Cantata No.140: Wachet auf, ruft uns die Stimme (27th Sunday after Trinity), BWV140: vii. Chorale: Gloria sei dir gesungen

## 引用

[^1]:“und die findigen Tiere merken es schon, daß wir nicht sehr verläßlich zu Haus sind in der gedeuteten Welt.”[Rainer Maria Rilke](https://en.wikipedia.org/wiki/Rainer_Maria_Rilke), [Duineser Elegien](https://site.douban.com/wingreading/widget/notes/7547565/note/580608016/): Die erste Elegie.[莱纳·玛利亚·里尔克](https://zh.wikipedia.org/wiki/%E8%8E%B1%E7%BA%B3%C2%B7%E7%8E%9B%E5%88%A9%E4%BA%9A%C2%B7%E9%87%8C%E5%B0%94%E5%85%8B)，[杜伊诺哀歌](https://site.douban.com/wingreading/widget/notes/7547565/note/580608016/)：第一挽歌
[^2]:[Ugolino_Martelli](https://music.163.com/#/user/home?id=257328437)原话
[^3]:[托马斯·内格尔](https://book.douban.com/author/115236), [你的第一本哲学书](https://book.douban.com/subject/26892991/)
[^4]:classical music（古典音乐），不同于古典主义音乐（也称维也纳古典乐派，1750-1825，from wikipedia）。关于古典音乐没有一个严格定义，但是有一些东西是大家公认的，一般不会将古典主义时期的音乐称为古典音乐，古典音乐包含巴洛克时期，古典主义时期，浪漫主义时期。我比较认可的古典音乐时期是wikipedia的[音乐史](https://zh.wikipedia.org/wiki/%E9%9F%B3%E6%A8%82%E5%8F%B2#20%E5%8F%8A21%E4%B8%96%E7%B4%80%E9%9F%B3%E6%A8%82%EF%BC%881901%E5%B9%B4%EF%BD%9E%E7%8F%BE%E5%9C%A8%EF%BC%89)中的1.4 西方艺术音乐与后来的先锋派，现代派。我主要听的曲目的也正是1.4 西方艺术音乐指的这段时期的音乐，主要有：少量的中世纪（格里高利圣咏）和文艺复兴时期，大量的巴洛克时期，古典主义时期，浪漫主义时期。中国民族音乐和其他地域民族音乐不算古典音乐，如果是遵循古典音乐传统（遵循特定曲式，结构等）而在当代或近现代编写的音乐我认为算古典音乐。词典中的定义：[classical music](https://www.lexico.com/definition/classical_music) 1 Serious music following long-established principles rather than a folk, jazz, or popular tradition. 1.1 (more specifically) music written in the European tradition during a period lasting approximately from 1750 to 1830, when forms such as the symphony, concerto, and sonata were standardized.

[^5]:“就算再怎么痛苦，就算等待着我们的是令人窒息的孤独的战斗，正因为有这样的喜悦，我们可以无数次的起身前行；在几百年前写下的音符，令生长国度、性别，眼睛的颜色都不相同的两人，想象出同样的声音；原本以为不能互相理解的人，只靠一个音符，就互相理解，互相吸引。”交响情人梦
[^6]:科普兰，[如何听懂音乐](https://book.douban.com/subject/26931364/)，P8
