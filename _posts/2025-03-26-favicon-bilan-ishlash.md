---
title: Faviconni Moslashtirish
author: cotes
date: 2025-03-26 02:02:00 +0500
categories: [Blogging, Tutorial]
tags: [favicon]
---

[**Chirpy**](https://github.com/cotes2020/jekyll-theme-chirpy/) ning [favicons](https://www.favicon-generator.org/about/) `assets/img/favicons/`{: .filepath} katalogida joylashgan. Siz ularni o'zingizning favikonlaringiz bilan almashtirishni xohlashingiz mumkin. Quyidagi bo'limlar sizga standart favikonlarni yaratish va almashtirish bo'yicha ko'rsatmalar beradi.

## Favicon yaratish

512x512 yoki undan katta o'lchamdagi kvadrat rasmni (PNG, JPG yoki SVG) tayyorlang va keyin onlayn vositaga [**Real Favicon Generator**](https://realfavicongenerator.net/) o'ting va <kbd>Select your Favicon image</kbd> tugmasini bosib, rasm faylingizni yuklang.

Keyingi bosqichda, veb-sahifa barcha foydalanish stsenariylarini ko'rsatadi. Siz standart variantlarni saqlashingiz mumkin, sahifaning pastki qismiga o'ting va <kbd>Generate your Favicons and HTML code</kbd> tugmasini bosib, favikonni yarating.

## Yuklab olish va almashtirish

Yaratilgan paketni yuklab oling, zip faylini oching va quyidagi ikkita faylni o'chiring:

- `browserconfig.xml`{: .filepath}
- `site.webmanifest`{: .filepath}

Keyin qolgan rasm fayllarini (`.PNG`{: .filepath} va `.ICO`{: .filepath}) Jekyll saytingizdagi `assets/img/favicons/`{: .filepath} katalogidagi asl fayllarni qoplash uchun nusxalab oling. Agar Jekyll saytingizda bu katalog hali mavjud bo'lmasa, uni yarating.

Quyidagi jadval favikon fayllaridagi o'zgarishlarni tushunishingizga yordam beradi:

| Fayl(lar)           | Onlayn vositadan                  | Chirpy'dan  |
|---------------------|:---------------------------------:|:-----------:|
| `*.PNG`             | ✓                                 | ✗           |
| `*.ICO`             | ✓                                 | ✗           |

<!-- markdownlint-disable-next-line -->
>  ✓ saqlashni anglatadi, ✗ o'chirishni anglatadi.
{: .prompt-info }

Keyingi safar saytni qurishda, favikon moslashtirilgan nashr bilan almashtiriladi.
