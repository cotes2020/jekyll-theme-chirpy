---
title: Yangi Maqola Yozish
author: cotes
date: 2025-03-26 02:04:00 +0500
categories: [Blogging, Tutorial]
tags: [writing]
render_with_liquid: false
---

Ushbu qo'llanma sizga _Chirpy_ shablonida qanday qilib maqola yozishni ko'rsatadi va agar siz ilgari Jekyll'dan foydalangan bo'lsangiz ham, ko'plab xususiyatlar uchun maxsus o'zgaruvchilarni o'rnatish kerak bo'lganligi sababli, uni o'qishga arziydi.

## Nomlash va Yo'l

`YYYY-MM-DD-TITLE.EXTENSION`{: .filepath} nomli yangi fayl yarating va uni root katalogidagi `_posts`{: .filepath} ichiga joylashtiring. Iltimos, `EXTENSION`{: .filepath} `md`{: .filepath} va `markdown`{: .filepath} dan biri bo'lishi kerakligini unutmang. Agar fayllarni yaratish vaqtini tejashni xohlasangiz, bu ishni bajarish uchun [`Jekyll-Compose`](https://github.com/jekyll/jekyll-compose) plaginidan foydalanishni ko'rib chiqing.

## Front Matter

Asosan, postning yuqori qismida quyidagi [Front Matter](https://jekyllrb.com/docs/front-matter/) ni to'ldirishingiz kerak:

```yaml
---
title: TITLE
date: YYYY-MM-DD HH:MM:SS +/-TTTT
categories: [TOP_CATEGORY, SUB_CATEGORY]
tags: [TAG]     # TAG nomlari har doim kichik harflarda bo'lishi kerak
---
```

> Postlarning _layout_ i sukut bo'yicha `post` ga o'rnatilgan, shuning uchun Front Matter blokida _layout_ o'zgaruvchisini qo'shish shart emas.
{: .prompt-tip }

### Sana Vaqt Zonasi

Postning chiqarilish sanasini aniq qayd etish uchun, siz nafaqat `_config.yml`{: .filepath} ning `timezone` ni o'rnatishingiz, balki Front Matter blokidagi `date` o'zgaruvchisida postning vaqt zonasini ham ko'rsatishingiz kerak. Format: `+/-TTTT`, masalan, `+0800`.

### Kategoriyalar va Teglar

Har bir postning `categories` ikkita elementgacha bo'lishi uchun mo'ljallangan va `tags` elementlar soni nol dan cheksizlikgacha bo'lishi mumkin. Masalan:

```yaml
---
categories: [Hayvon, Hasharot]
tags: [ari]
---
```

### Muallif Ma'lumotlari

Postning muallif ma'lumotlari odatda _Front Matter_ da to'ldirilishi shart emas, ular sukut bo'yicha konfiguratsiya faylidagi `social.name` va `social.links` ning birinchi yozuvidan olinadi. Ammo siz uni quyidagicha o'zgartirishingiz mumkin:

`_data/authors.yml` ga muallif ma'lumotlarini qo'shish (Agar veb-saytingizda bu fayl bo'lmasa, uni yaratishdan tortinmang).

```yaml
<author_id>:
  name: <to'liq ism>
  twitter: <muallifning twitteri>
  url: <muallifning veb-sahifasi>
```
{: file="_data/authors.yml" }

Keyin `author` ni bitta yozuvni yoki `authors` ni bir nechta yozuvlarni ko'rsatish uchun ishlating:

```yaml
---
author: <author_id>                     # bitta yozuv uchun
# yoki
authors: [<author1_id>, <author2_id>]   # bir nechta yozuvlar uchun
---
```

Shuni aytib o'tish kerakki, `author` kaliti bir nechta yozuvlarni ham aniqlay oladi.

> Muallif ma'lumotlarini `_data/authors.yml`{: .filepath } faylidan o'qishning foydasi shundaki, sahifada `twitter:creator` meta tegi bo'ladi, bu esa [Twitter Cards](https://developer.twitter.com/en/docs/twitter-for-websites/cards/guides/getting-started#card-and-content-attribution) ni boyitadi va SEO uchun foydalidir.
{: .prompt-info }

### Post Tavsifi

Sukut bo'yicha, postning birinchi so'zlari bosh sahifada postlar ro'yxati uchun, _Further Reading_ bo'limida va RSS ozuqasining XML da ko'rsatiladi. Agar post uchun avtomatik yaratilgan tavsifni ko'rsatishni xohlamasangiz, uni quyidagicha _Front Matter_ da `description` maydonidan foydalanib sozlashingiz mumkin:

```yaml
---
description: Postning qisqa mazmuni.
---
```

Bundan tashqari, `description` matni post sahifasida post sarlavhasi ostida ham ko'rsatiladi.

## Mazmuni Jadvali

Sukut bo'yicha, postning o'ng panelida **M**azmuni **J**advali (TOC) ko'rsatiladi. Agar uni global miqyosda o'chirmoqchi bo'lsangiz, `_config.yml`{: .filepath} ga o'ting va `toc` o'zgaruvchisini `false` ga o'rnating. Agar TOC ni ma'lum bir post uchun o'chirmoqchi bo'lsangiz, postning [Front Matter](https://jekyllrb.com/docs/front-matter/) ga quyidagini qo'shing:

```yaml
---
toc: false
---
```

## Izohlar

Izohlar uchun global sozlama `_config.yml`{: .filepath} faylidagi `comments.provider` opsiyasi bilan belgilanadi. Ushbu o'zgaruvchi uchun izoh tizimi tanlanganidan so'ng, barcha postlar uchun izohlar yoqiladi.

Agar ma'lum bir post uchun izohni yopmoqchi bo'lsangiz, postning **Front Matter** ga quyidagini qo'shing:

```yaml
---
comments: false
---
```

## Media

Biz _Chirpy_ da rasmlar, audio va videolarni media resurslari deb ataymiz.

### URL Prefiksi

Ba'zan biz postdagi bir nechta resurslar uchun takroriy URL prefikslarini belgilashimiz kerak bo'ladi, bu zerikarli vazifani `cdn` va `media_subpath` parametrlarini o'rnatish orqali oldini olish mumkin.

- Agar siz media fayllarni joylashtirish uchun CDN dan foydalansangiz, `_config.yml`{: .filepath } da `cdn` ni belgilashingiz mumkin. Sayt avatar va postlar uchun media resurslarining URL lari CDN domen nomi bilan prefiks qilinadi.

  ```yaml
  cdn: https://cdn.com
  ```
  {: file='_config.yml' .nolineno }

- Joriy post/sahifa diapazoni uchun resurs yo'li prefiksini belgilash uchun, postning _front matter_ da `media_subpath` ni o'rnating:

  ```yaml
  ---
  media_subpath: /path/to/media/
  ---
  ```
  {: .nolineno }

`site.cdn` va `page.media_subpath` opsiyalari alohida yoki birgalikda ishlatilishi mumkin, bu esa yakuniy resurs URL ni moslashuvchan tarzda tuzishga imkon beradi: `[site.cdn/][page.media_subpath/]file.ext`

### Rasmlar

#### Sarlavha

Rasmning keyingi qatoriga kursiv qo'shing, keyin u sarlavha bo'lib, rasmning pastki qismida paydo bo'ladi:

```markdown
![img-description](/path/to/image)
_Rasm Sarlavhasi_
```
{: .nolineno}

#### O'lcham

Rasm yuklanayotganda sahifa tarkibining tartibini o'zgartirmaslik uchun har bir rasm uchun kenglik va balandlikni o'rnatishimiz kerak.

```markdown
![Desktop View](/assets/img/sample/mockup.png){: width="700" height="400" }
```
{: .nolineno}

> SVG uchun, hech bo'lmaganda uning _width_ ni ko'rsatishingiz kerak, aks holda u ko'rsatilmaydi.
{: .prompt-info }

_Chirpy v5.0.0_ dan boshlab, `height` va `width` qisqartmalarni qo'llab-quvvatlaydi (`height` → `h`, `width` → `w`). Quyidagi misol yuqoridagi bilan bir xil ta'sirga ega:

```markdown
![Desktop View](/assets/img/sample/mockup.png){: w="700" h="400" }
```
{: .nolineno}

#### Pozitsiya

Sukut bo'yicha, rasm markazda joylashgan, lekin siz `normal`, `left` va `right` sinflaridan birini ishlatib pozitsiyani belgilashingiz mumkin.

> Pozitsiya belgilanganidan so'ng, rasm sarlavhasi qo'shilmasligi kerak.
{: .prompt-warning }

- **Normal pozitsiya**

  Quyidagi misolda rasm chapga hizalanadi:

  ```markdown
  ![Desktop View](/assets/img/sample/mockup.png){: .normal }
  ```
  {: .nolineno}

- **Chapga suzuvchi**

  ```markdown
  ![Desktop View](/assets/img/sample/mockup.png){: .left }
  ```
  {: .nolineno}

- **O'ngga suzuvchi**

  ```markdown
  ![Desktop View](/assets/img/sample/mockup.png){: .right }
  ```
  {: .nolineno}

#### Qorong'i/Yorug' rejim

Siz rasmlarni qorong'i/yorug' rejimda mavzu afzalliklariga moslashtirishingiz mumkin. Bu uchun siz ikkita rasm tayyorlashingiz kerak, biri qorong'i rejim uchun, ikkinchisi yorug' rejim uchun, keyin ularga maxsus sinf (`dark` yoki `light`) tayinlang:

```markdown
![Faqat yorug' rejim](/path/to/light-mode.png){: .light }
![Faqat qorong'i rejim](/path/to/dark-mode.png){: .dark }
```

#### So'ya

Dastur oynasining skrinshotlari so'ya effektini ko'rsatish uchun ko'rib chiqilishi mumkin:

```markdown
![Desktop View](/assets/img/sample/mockup.png){: .shadow }
```
{: .nolineno}

#### Oldindan Ko'rish Rasm

Agar postning yuqori qismiga rasm qo'shmoqchi bo'lsangiz, iltimos, `1200 x 630` o'lchamdagi rasmni taqdim eting. Iltimos, agar rasmning tomonlar nisbati `1.91 : 1` ga mos kelmasa, rasm o'lchamlanadi va kesiladi.

Ushbu talablarni bilgan holda, rasmning atributini sozlashni boshlashingiz mumkin:

```yaml
---
image:
  path: /path/to/image
  alt: rasmning muqobil matni
---
```

Shuni yodda tutingki, [`media_subpath`](#url-prefix) oldindan ko'rish rasmiga ham o'tkazilishi mumkin, ya'ni u o'rnatilganda, `path` atributi faqat rasm fayl nomini talab qiladi.

Oddiy foydalanish uchun, siz `image` ni faqat yo'lni belgilash uchun ishlatishingiz mumkin.

```yml
---
image: /path/to/image
---
```

#### LQIP

Oldindan ko'rish rasmlari uchun:

```yaml
---
image:
  lqip: /path/to/lqip-file # yoki base64 URI
---
```

> Siz LQIP ni "[Matn va Tipografiya](../text-and-typography/)" postining oldindan ko'rish rasmida kuzatishingiz mumkin.

Oddiy rasmlar uchun:

```markdown
![Rasm tavsifi](/path/to/image){: lqip="/path/to/lqip-file" }
```
{: .nolineno }

### Video

#### Ijtimoiy Media Platformasi

Siz quyidagi sintaksis bilan ijtimoiy media platformalaridan videolarni joylashtirishingiz mumkin:

```liquid
{% include embed/{Platform}.html id='{ID}' %}
```

Bu yerda `Platform` platforma nomining kichik harflari, va `ID` video ID si.

Quyidagi jadvalda berilgan video URL da kerakli ikkita parametrni qanday olish mumkinligi ko'rsatilgan va siz hozirda qo'llab-quvvatlanadigan video platformalarini ham bilib olishingiz mumkin.

| Video URL                                                                                          | Platforma   | ID             |
| -------------------------------------------------------------------------------------------------- | ---------- | :------------- |
| [https://www.**youtube**.com/watch?v=**H-B46URT4mg**](https://www.youtube.com/watch?v=H-B46URT4mg) | `youtube`  | `H-B46URT4mg`  |
| [https://www.**twitch**.tv/videos/**1634779211**](https://www.twitch.tv/videos/1634779211)         | `twitch`   | `1634779211`   |
| [https://www.**bilibili**.com/video/**BV1Q44y1B7Wf**](https://www.bilibili.com/video/BV1Q44y1B7Wf) | `bilibili` | `BV1Q44y1B7Wf` |

#### Video Fayllar

Agar siz video faylni to'g'ridan-to'g'ri joylashtirmoqchi bo'lsangiz, quyidagi sintaksisdan foydalaning:

```liquid
{% include embed/video.html src='{URL}' %}
```

Bu yerda `URL` video faylga URL, masalan, `/path/to/sample/video.mp4`.

Siz joylashtirilgan video fayl uchun qo'shimcha atributlarni ham belgilashingiz mumkin. Bu yerda ruxsat etilgan atributlarning to'liq ro'yxati keltirilgan.

- `poster='/path/to/poster.png'` — video yuklanayotganda ko'rsatiladigan poster rasm
- `title='Matn'` — video uchun sarlavha, video ostida ko'rsatiladi va rasmlar uchun bir xil ko'rinishga ega
- `autoplay=true` — video imkon qadar tezroq avtomatik ravishda ijro etishni boshlaydi
- `loop=true` — video oxiriga yetganda avtomatik ravishda boshiga qaytadi
- `muted=true` — audio dastlab o'chiriladi
- `types` — qo'shimcha video formatlarining kengaytmalarini `|` bilan ajratib ko'rsating. Ushbu fayllar asosiy video faylingiz bilan bir xil katalogda mavjudligiga ishonch hosil qiling.

Quyidagi misol yuqoridagi barcha atributlardan foydalanishni ko'rsatadi:

```liquid
{%
  include embed/video.html
  src='/path/to/video.mp4'
  types='ogg|mov'
  poster='poster.png'
  title='Demo video'
  autoplay=true
  loop=true
  muted=true
%}
```

### Audios

Agar siz audio faylni to'g'ridan-to'g'ri joylashtirmoqchi bo'lsangiz, quyidagi sintaksisdan foydalaning:

```liquid
{% include embed/audio.html src='{URL}' %}
```

Bu yerda `URL` audio faylga URL, masalan, `/path/to/audio.mp3`.

Siz joylashtirilgan audio fayl uchun qo'shimcha atributlarni ham belgilashingiz mumkin. Bu yerda ruxsat etilgan atributlarning to'liq ro'yxati keltirilgan.

- `title='Matn'` — audio uchun sarlavha, audio ostida ko'rsatiladi va rasmlar uchun bir xil ko'rinishga ega
- `types` — qo'shimcha audio formatlarining kengaytmalarini `|` bilan ajratib ko'rsating. Ushbu fayllar asosiy audio faylingiz bilan bir xil katalogda mavjudligiga ishonch hosil qiling.

Quyidagi misol yuqoridagi barcha atributlardan foydalanishni ko'rsatadi:

```liquid
{%
  include embed/audio.html
  src='/path/to/audio.mp3'
  types='ogg|wav|aac'
  title='Demo audio'
%}
```

## Mahkamlangan Postlar

Siz bir yoki bir nechta postlarni bosh sahifaning yuqori qismiga mahkamlashingiz mumkin va mahkamlangan postlar chiqarilish sanasiga ko'ra teskari tartibda saralanadi. Yoqish uchun:

```yaml
---
pin: true
---
```

## Ko'rsatmalar

Bir nechta ko'rsatma turlari mavjud: `tip`, `info`, `warning`, va `danger`. Ular `prompt-{type}` sinfini blockquote ga qo'shish orqali yaratilishi mumkin. Masalan, `info` turidagi ko'rsatmani quyidagicha aniqlang:

```md
> Ko'rsatma uchun misol satri.
{: .prompt-info }
```
{: .nolineno }

## Sintaksis

### Inline Kod

```md
`inline kod qismi`
```
{: .nolineno }

### Fayl Yo'lini Ta'kidlash

```md
`/path/to/a/file.extend`{: .filepath}
```
{: .nolineno }

### Kod Bloki

Markdown belgilar ```` ``` ```` quyidagicha kod blokini osongina yaratishi mumkin:

````md
```
Bu oddiy matn kod parchasidir.
```
````

#### Tilni Belgilash

```` ```{language} ```` dan foydalanib, sintaksisni ta'kidlash bilan kod blokini olasiz:

````markdown
```yaml
key: value
```
````

> Jekyll tegi `{% highlight %}` ushbu mavzu bilan mos kelmaydi.
{: .prompt-danger }

#### Qator Raqami

Sukut bo'yicha, `plaintext`, `console` va `terminal` dan tashqari barcha tillar qator raqamlarini ko'rsatadi. Agar kod blokining qator raqamini yashirmoqchi bo'lsangiz, unga `nolineno` sinfini qo'shing:

````markdown
```shell
echo 'Qator raqamlari yo'q!'
```
{: .nolineno }
````

#### Fayl Nomini Belgilash

Siz kod tilining kod blokining yuqori qismida ko'rsatilishini payqagan bo'lishingiz mumkin. Agar uni fayl nomi bilan almashtirmoqchi bo'lsangiz, atribut `file` ni qo'shishingiz mumkin:

````markdown
```shell
# content
```
{: file="path/to/file" }
````

#### Liquid Kodlari

Agar **Liquid** parchasini ko'rsatmoqchi bo'lsangiz, liquid kodini `{% raw %}` va `{% endraw %}` bilan o'rab oling:

````markdown
{% raw %}
```liquid
{% if product.title contains 'Pack' %}
  This product's title contains the word Pack.
{% endif %}
```
{% endraw %}
````

Yoki postning YAML blokiga `render_with_liquid: false` (Jekyll 4.0 yoki undan yuqori versiya talab qilinadi) ni qo'shing.

## Matematika

Biz matematikani yaratish uchun [**MathJax**][mathjax] dan foydalanamiz. Veb-saytning ishlash sabablari uchun matematik xususiyat sukut bo'yicha yuklanmaydi. Ammo uni quyidagicha yoqish mumkin:

[mathjax]: https://www.mathjax.org/

```yaml
---
math: true
---
```

Matematik xususiyatni yoqqandan so'ng, quyidagi sintaksis bilan matematik tenglamalarni qo'shishingiz mumkin:

- **Blok matematikasi** `$$ math $$` bilan qo'shilishi kerak, `$$` oldidan va keyin **majburiy** bo'sh satrlar bo'lishi kerak
  - **Tenglama raqamini kiritish** `$$\begin{equation} math \end{equation}$$` bilan qo'shilishi kerak
  - **Tenglama raqamini keltirish** tenglama blokida `\label{eq:label_name}` va matn bilan bir qatorda `\eqref{eq:label_name}` bilan amalga oshirilishi kerak (quyidagi misolga qarang)
- **Inline matematikasi** (qatorda) `$$ math $$` bilan qo'shilishi kerak, `$$` oldidan yoki keyin hech qanday bo'sh satr bo'lmasdan
- **Inline matematikasi** (ro'yxatlarda) birinchi `$` ni qochirish bilan qo'shilishi kerak

```markdown
<!-- Blok matematikasi, barcha bo'sh satrlarni saqlang -->

$$
LaTeX_math_expression
$$

<!-- Tenglama raqamlash, barcha bo'sh satrlarni saqlang  -->

$$
\begin{equation}
  LaTeX_math_expression
  \label{eq:label_name}
\end{equation}
$$

Can be referenced as \eqref{eq:label_name}.

<!-- Inline matematikasi qatorlarda, bo'sh satrlar yo'q -->

"Lorem ipsum dolor sit amet, $$ LaTeX_math_expression $$ consectetur adipiscing elit."

<!-- Inline matematikasi ro'yxatlarda, birinchi `$` ni qochiring -->

1. \$$ LaTeX_math_expression $$
2. \$$ LaTeX_math_expression $$
3. \$$ LaTeX_math_expression $$
```

> `v7.0.0` dan boshlab, **MathJax** uchun konfiguratsiya opsiyalari `assets/js/data/mathjax.js`{: .filepath } fayliga ko'chirildi va siz kerak bo'lganda opsiyalarni o'zgartirishingiz mumkin, masalan, [kengaytmalar][mathjax-exts] qo'shish.  
> Agar siz saytni `chirpy-starter` orqali qurayotgan bo'lsangiz, o'sha faylni gem o'rnatish katalogidan (buyruq `bundle info --path jekyll-theme-chirpy` bilan tekshiring) repozitoriyingizdagi bir xil katalogga nusxalash.
{: .prompt-tip }

[mathjax-exts]: https://docs.mathjax.org/en/latest/input/tex/extensions/index.html

## Mermaid

[**Mermaid**](https://github.com/mermaid-js/mermaid) ajoyib diagramma yaratish vositasi. Uni postingizda yoqish uchun YAML blokiga quyidagini qo'shing:

```yaml
---
mermaid: true
---
```

Keyin uni boshqa markdown tillari kabi ishlatishingiz mumkin: grafik kodini ```` ```mermaid ```` va ```` ``` ```` bilan o'rab oling.

## Ko'proq Bilish

Jekyll postlari haqida ko'proq ma'lumot olish uchun [Jekyll Docs: Posts](https://jekyllrb.com/docs/posts/) ni ko'rib chiqing.
