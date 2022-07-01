---
title: Anonymous Redirect / Referrer
date: 2022-06-23 12:10:00 +0200
categories: Project
tags: javascript security privacy
image: 
---

Anonim linkler oluşturmak için basit HTML + JS kodu.

Orijinal olarak [AdGuard](https://github.com/HuzunluArtemis/AnonymousRedirect) tarafından yazıldı. Ben biraz düzenledim.

Internet Explorer 11'de çalışmayabilir.

## Link Oluşturma

`https://huzunluartemis.github.io/AnonymousRedirect/redirect.html?url=[Buraya Linkin Gelecek]`

## Örnekler

- `https://huzunluartemis.github.io/AnonymousRedirect/redirect.html?url=https://github.com/HuzunluArtemis`
- Uyarı: Gerçek boşluk kullanma. Linkinde boşluk varsa yanlış yapıyorsun demektir. Bunun için tarayıcının url kısmından kopyalamanı öneririm. Orası doğru şekilde kopyalayacaktır. Tarayıcın yapamıyorsa doğru url oluşturmak için [şurayı](https://www.urlencoder.org/) kullanabilirsin.
- Doğru Kullanım:
    - `https://huzunluartemis.github.io/AnonymousRedirect/redirect.html?url=https://www.google.com/search?q=google%20search%20console`
- Yanlış Kullanım:
    - `https://huzunluartemis.github.io/AnonymousRedirect/redirect.html?url=https://www.google.com/search?q=url encode decode nedir`
    - Doğrusu: `https://huzunluartemis.github.io/AnonymousRedirect/redirect.html?url=https://www.google.com/search?q=url+encode+decode+nedir`
    - Neden böyle olduğunu anlamak mı [istiyorsun?](https://huzunluartemis.github.io/AnonymousRedirect/redirect.html?url=https://www.google.com/search?q=url+encode+decode+nedir)
- Programlama dillerinde de bunu yapmalısın. Bu siteyi api gibi kullanabilirsin. Birkaç programlama dilinden örnekler [buldum](https://huzunluartemis.github.io/AnonymousRedirect/redirect.html?url=https://www.urlencoder.io/blog/). Kolay gelsin.

## Lisans

![](https://www.gnu.org/graphics/gplv3-127x51.png)

You can use, study share and improve it at your will. Specifically you can redistribute and/or modify it under the terms of the [GNU General Public License](https://www.gnu.org/licenses/gpl-3.0.html) as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.