---
title: Boshlash Uchun Kerakli Tushunchalar
description: >-
  Chirpy asoslari bilan tanishing. Ushbu qo'llanmada siz Chirpy asosidagi veb-saytni qanday o'rnatish, sozlash va ishlatishni, shuningdek, uni veb-serverga joylashtirishni o'rganasiz.
author: cotes
date: 2025-03-26 02:01:00 +0500
categories: [Blogging, Tutorial]
tags: [boshlash]
pin: true
media_subpath: '/posts/20180809'
---

## Sayt Repozitoriyasini Yaratish

Sayt repozitoriyasini yaratishda, ehtiyojlaringizga qarab ikkita variant mavjud:

### Variant 1. Starterdan Foydalanish (Tavsiya etiladi)

Ushbu yondashuv yangilanishlarni soddalashtiradi, keraksiz fayllarni ajratadi va minimal sozlash bilan yozishga e'tibor qaratmoqchi bo'lgan foydalanuvchilar uchun juda mos keladi.

1. GitHub'ga kiring va [**starter**][starter] sahifasiga o'ting.
2. <kbd>Use this template</kbd> tugmasini bosing va keyin <kbd>Create a new repository</kbd> ni tanlang.
3. Yangi repozitoriyani `<username>.github.io` deb nomlang, bu yerda `username` sizning kichik harflardagi GitHub foydalanuvchi nomingiz bilan almashtiriladi.

### Variant 2. Mavzuni Forklash

Ushbu yondashuv xususiyatlarni yoki UI dizaynini o'zgartirish uchun qulay, lekin yangilanishlar paytida qiyinchiliklar tug'diradi. Shuning uchun, agar siz Jekyll bilan tanish bo'lmasangiz va ushbu mavzuni jiddiy o'zgartirishni rejalashtirmasangiz, bu usulni sinab ko'rmang.

1. GitHub'ga kiring.
2. [Mavzu repozitoriyasini fork qiling](https://github.com/cotes2020/jekyll-theme-chirpy/fork).
3. Yangi repozitoriyani `<username>.github.io` deb nomlang, bu yerda `username` sizning kichik harflardagi GitHub foydalanuvchi nomingiz bilan almashtiriladi.

## Muhitni Sozlash

Repozitoriya yaratilgandan so'ng, rivojlanish muhitini sozlash vaqti keldi. Buning ikkita asosiy usuli mavjud:

### Dev Containers'dan Foydalanish (Windows uchun Tavsiya etiladi)

Dev Containers Docker yordamida izolyatsiyalangan muhitni taqdim etadi, bu sizning tizimingiz bilan ziddiyatlarni oldini oladi va barcha bog'liqliklarni konteyner ichida boshqaradi.

**Qadamlar**:

1. Docker'ni o'rnating:
   - Windows/macOS'da [Docker Desktop][docker-desktop] ni o'rnating.
   - Linux'da [Docker Engine][docker-engine] ni o'rnating.
2. [VS Code][vscode] va [Dev Containers kengaytmasi][dev-containers] ni o'rnating.
3. Repozitoriyangizni klonlang:
   - Docker Desktop uchun: VS Code'ni ishga tushiring va [repozitoriyangizni konteyner hajmida klonlang][dc-clone-in-vol].
   - Docker Engine uchun: Repozitoriyangizni lokal ravishda klonlang, keyin VS Code orqali [konteynerda oching][dc-open-in-container].
4. Dev Containers sozlamalari tugashini kuting.

### Mahalliy Sozlash (Unix-like OS uchun Tavsiya etiladi)

Unix-like tizimlar uchun, siz muhitni mahalliy ravishda optimal ishlash uchun sozlashingiz mumkin, lekin Dev Containers'dan alternativ sifatida ham foydalanishingiz mumkin.

**Qadamlar**:

1. [Jekyll o'rnatish qo'llanmasi](https://jekyllrb.com/docs/installation/) ga amal qilib Jekyll'ni o'rnating va [Git](https://git-scm.com/) o'rnatilganligiga ishonch hosil qiling.
2. Repozitoriyangizni lokal mashinangizga klonlang.
3. Agar mavzuni fork qilgan bo'lsangiz, [Node.js][nodejs] ni o'rnating va repozitoriyaning root katalogida `bash tools/init.sh` ni ishga tushiring.
4. Repozitoriyaning root katalogida `bundle` buyrug'ini ishga tushiring va bog'liqliklarni o'rnating.

## Foydalanish

### Jekyll Serverni Ishga Tushirish

Saytni lokal ravishda ishga tushirish uchun quyidagi buyruqdan foydalaning:

```terminal
$ bundle exec jekyll serve
```

> Agar siz Dev Containers'dan foydalanayotgan bo'lsangiz, bu buyruqni **VS Code** Terminalida ishga tushirishingiz kerak.
{: .prompt-info }

Bir necha soniyadan so'ng, lokal server <http://127.0.0.1:4000> da mavjud bo'ladi.

### Sozlash

`_config.yml`{: .filepath} dagi o'zgaruvchilarni kerakli tarzda yangilang. Ba'zi odatiy variantlar quyidagilarni o'z ichiga oladi:

- `url`
- `avatar`
- `timezone`
- `lang`

### Ijtimoiy Aloqa Variantlari

Ijtimoiy aloqa variantlari yon panelning pastki qismida ko'rsatiladi. Siz `_data/contact.yml`{: .filepath} faylida ma'lum aloqa variantlarini yoqishingiz yoki o'chirishingiz mumkin.

### Stilni Moslashtirish

Stilni moslashtirish uchun, mavzuning `assets/css/jekyll-theme-chirpy.scss`{: .filepath} faylini Jekyll saytingizdagi xuddi shu yo'lga nusxalab oling va fayl oxirida o'zingizning maxsus stillaringizni qo'shing.

### Statik Aktivlarni Moslashtirish

Statik aktivlar konfiguratsiyasi `5.1.0` versiyasida joriy etilgan. Statik aktivlarning CDN `_data/origin/cors.yml`{: .filepath} da aniqlangan. Siz veb-saytingiz joylashtirilgan hududdagi tarmoq sharoitlariga qarab ularning ba'zilarini almashtirishingiz mumkin.

Agar statik aktivlarni o'zingiz joylashtirishni afzal ko'rsangiz, [_chirpy-static-assets_](https://github.com/cotes2020/chirpy-static-assets#readme) repozitoriyasiga murojaat qiling.

## Joylashtirish

Joylashtirishdan oldin, `_config.yml`{: .filepath} faylini tekshiring va `url` to'g'ri sozlanganligiga ishonch hosil qiling. Agar siz [**loyiha sayti**](https://help.github.com/en/github/working-with-github-pages/about-github-pages#types-of-github-pages-sites) ni afzal ko'rsangiz va maxsus domen ishlatmasangiz yoki GitHub Pages'dan boshqa veb-serverda veb-saytingizga baza URL bilan tashrif buyurmoqchi bo'lsangiz, `baseurl` ni loyihangiz nomiga, masalan, `/project-name` ga o'rnatishni unutmang.

Endi Jekyll saytingizni joylashtirish uchun quyidagi usullardan _BIRINI_ tanlashingiz mumkin.

### GitHub Actions yordamida joylashtirish

Quyidagilarni tayyorlang:

- Agar siz GitHub Free rejasida bo'lsangiz, sayt repozitoriyangizni ommaviy saqlang.
- Agar `Gemfile.lock`{: .filepath} faylini repozitoriyaga qo'shgan bo'lsangiz va lokal mashinangiz Linuxda ishlamasa, lock faylining platforma ro'yxatini yangilang:

  ```console
  $ bundle lock --add-platform x86_64-linux
  ```

Keyin _Pages_ xizmatini sozlang:

1. GitHub'dagi repozitoriyangizga o'ting. _Settings_ yorlig'ini tanlang, so'ng chap navigatsiya panelida _Pages_ ni bosing. **Source** bo'limida ( _Build and deployment_ ostida), ochiladigan menyudan [**GitHub Actions**][pages-workflow-src] ni tanlang.  
   ![Build source](pages-source-light.png){: .light .border .normal w='375' h='140' }
   ![Build source](pages-source-dark.png){: .dark .normal w='375' h='140' }

2. Har qanday commitni GitHub'ga yuboring, bu _Actions_ ish jarayonini ishga tushiradi. Repozitoriyaning _Actions_ yorlig'ida _Build and Deploy_ ish jarayonini ko'rishingiz kerak. Qurilish tugallangandan va muvaffaqiyatli bo'lgandan so'ng, sayt avtomatik ravishda joylashtiriladi.

Endi saytga kirish uchun GitHub tomonidan taqdim etilgan URL manziliga tashrif buyurishingiz mumkin.

### Qo'lda Qurish va Joylashtirish

O'zingiz joylashtirgan serverlar uchun, saytni lokal mashinangizda qurishingiz va keyin sayt fayllarini serverga yuklashingiz kerak bo'ladi.

Loyiha manbasi root katalogiga o'ting va saytni quyidagi buyruq bilan qurib chiqing:

```console
$ JEKYLL_ENV=production bundle exec jekyll b
```

Agar chiqish yo'lini belgilamagan bo'lsangiz, yaratilgan sayt fayllari loyiha root katalogidagi `_site`{: .filepath} papkasiga joylashtiriladi. Ushbu fayllarni maqsadli serverga yuklang.

[nodejs]: https://nodejs.org/
[starter]: https://github.com/cotes2020/chirpy-starter
[pages-workflow-src]: https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site#publishing-with-a-custom-github-actions-workflow
[docker-desktop]: https://www.docker.com/products/docker-desktop/
[docker-engine]: https://docs.docker.com/engine/install/
[vscode]: https://code.visualstudio.com/
[dev-containers]: https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers
[dc-clone-in-vol]: https://code.visualstudio.com/docs/devcontainers/containers#_quick-start-open-a-git-repository-or-github-pr-in-an-isolated-container-volume
[dc-open-in-container]: https://code.visualstudio.com/docs/devcontainers/containers#_quick-start-open-an-existing-folder-in-a-container
