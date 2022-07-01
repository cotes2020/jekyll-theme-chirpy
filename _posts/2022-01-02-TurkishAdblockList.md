---
title: TurkishAdblockList
date: 2022-01-02 12:10:00 +0200
categories: Project
tags: security
image: 
---

# Kullanım ve Uyarı

AdAway ile kullanınız. -çoğunlukla- Android için tasarlanmıştır. Daha önceden Nano Adblocker tavsiye etmiştim. Nano adblocker artık bir virüs olduğu için cihazlarınızdan kaldırın ve ublock origin yükleyin! Ana depo [burada](https://github.com/HuzunluArtemis/TurkishAdblockList)

## Windows/Linux/Mac için Önerilen

- İnternet tarayıcınızda [ublock origin](https://github.com/gorhill/uBlock) kullanarak etkin bir şekilde element filtrelemesi yapabilirsiniz. Bunun için öncelikle tarayıcınız için olanını edinin: [Chrome için](https://chrome.google.com/webstore/detail/ublock-origin/cjpalhdlnbpafiamejdnhcphjbkeiagm?hl=tr) - 
[Firefox için](https://addons.mozilla.org/tr/firefox/addon/ublock-origin/) - [Edge için](https://microsoftedge.microsoft.com/addons/detail/ublock-origin/odfafepnkmbhccpbejgmiehpchacaeak) - [Opera için](https://microsoftedge.microsoft.com/addons/detail/ublock-origin/odfafepnkmbhccpbejgmiehpchacaeak)

- Ardından **ublock origin > Kontrol Paneli > Süzgeç Listeleri > İçe Aktar (en altta)** bölümünden açılan url ekleme kutusuna şu linkleri ekleyin:

```
https://raw.githubusercontent.com/huzunluartemis/TurkishAdblockList/main/src/HostsList.txt
https://raw.githubusercontent.com/huzunluartemis/TurkishAdblockList/main/src/BadIpList.txt
https://raw.githubusercontent.com/huzunluartemis/TurkishAdblockList/main/src/ElementalList.txt
https://raw.githubusercontent.com/rampageX/fuckfuckadblock/master/my_antiadblock_selection.txt
https://zerodot1.gitlab.io/CoinBlockerLists/hosts_browser
```

![bunun gibi](https://raw.githubusercontent.com/HuzunluArtemis/TurkishAdblockList/main/docs/bilgi1.png)

- Ardından sağ üstte beliren **"Değişiklikleri Kaydet"** düğmesine tıklayarak sayfadan çıkabilirsiniz. 
- Artık ayarlar sayfasına girip **"Güncelle"** tuşuna basmanıza gerek yok. Liste diğer listelerin yaptığı gibi belli aralıklarla kendini güncelleyecektir. 

### Ayarlamalarınız Bittiğinde Şöyle Görünmelidir:

![bunun gibi](https://raw.githubusercontent.com/HuzunluArtemis/TurkishAdblockList/main/docs/bilgi2.png)

## Android için Seçenekler

Ücretsiz yazılımlar genelde hosts seviyesinde reklam engelleyebilmekte ve çoğu zaman uygulamalarda reklamları engelleyememektedir. 

### Android için AdGuard // ücretli - cihaz kök-erişimli değilse

Android için AdGuard yazılımı bu konuda en iyisi denilebilir. Ücretli (Premium) sürümde birçok filtre kullanma imkanı size sunuyor ve daha gelişmiş bir reklam engelleme teknolojisi kullanıyor. Dediğim gibi bu uygulama ücretlidir ve [Google reklam politikalarının işine gelmedeği için Google Play'den kaldırılmıştır](https://blog.adguard.com/en/google-removes-adguard-android-app-google-play/).
- [Android için AdGuard](https://adguard.com/tr/adguard-android/overview.html)
- Kurulum ve kullanımı kolaydır, ROOT gerektirmez.
- 14 gün ücretsiz tam sürümü deneyebilirsiniz,
- Ayarlardan istediğiniz filtreleri (yukarıdaki filtrelerin aynılarını) etkinleştirebilirsiniz.
- Filtreleme yöntemini "Yüksek Kaliteli" yapın.
- HTTPS kullanan reklam ağlarını ve uygulamaları (Youtube reklamları gibi) engelleyebilir.

### DNS66 // ücretsiz - cihaz kök-erişimli değilse

Telefonunuzda root işlemi yapmak zor ve riskli olabilir. Telefonunuzu garanti kapsamı dışına çıkarabilir. Telefonunuz root edilmemiş ise aşağıdaki adımlarla reklamları engelleyebilirsiniz.

- [DNS66](https://github.com/julian-klode/dns66/releases) uygulamasını (Assets kısmındaki ".apk" uzantılı ve her zaman en üsttteki dosyayı) indirin.
- Bilinmeyen kaynaklar uyarısına izin verin. Uygulamayı telefonunuza / tabletinize kurun.
- Uygulamayı açın, alt bölümde "Domain Filters" sekmesine dokunun.
- Sağ alt taraftaki artı (artı) ikonuna dokunun ve aşağıdaki değerleri yazın.
- Title: huzunluartemis TurkishAdblockList
- Location: ```https://raw.githubusercontent.com/huzunluartemis/TurkishAdblockList/main/src/HostsList.txt```
- Action: Deny
- Sağ üstten "Save" diyerek bu ayarları kaydedin. Yukarıdaki yenile butonuna dokunarak güncellemeleri indirin.
- Start/Stop menüsüne geçin, ekrana uzunca dokunun ve filtrelemeyi etkinleştirin.
- Eğer bildirim alanında anahtar işareti görüyorsanız, filtrelerimiz etkindir. Reklamsız gezinebilirsiniz.

### AdAway (Root) // ücretsiz, önerilen, şarj dostu, cihaz kök-erişimliyse

Root erişim izniniz varsa telefonun kendi "hosts" dosyasını değiştirmelisiniz. Bu, batarya ve RAM tasarrufu sağlar.
Kök erişim (root) izniniz varsa AdAway uygulamasını kullanabilirsiniz. Host dosyaları ile reklam engelleyen ücretsiz bir uygulamadır.

- AdAway uygulamasını [buradan](https://github.com/AdAway/AdAway/releases) (Assets kısmındaki ".apk" uzantılı ve her zaman en üsttteki dosyayı) indirin.
- Uygulamayı telefonunuza veya tabletinize kurun.
- Uygulamayı açın ve uygulama menüsünden "Host kaynakları" sekmesini açın.
- Sağ üst köşedeki '+' işaretine dokunun. Bir bağlantı girmeniz istenecektir.
- Aşağıdaki adreslerden **olmayanları** kopyalayıp bu kısma yapıştırın ve ekleyin.
- (Bunlar benim önerilerim ve normal Android kullanıcıları için yeterlidir. Ben daha fazla istiyorum diyorsanız [şuraya](https://github.com/huzunluartemis/TurkishAdblockList/blob/master/OTHERS.md) bakabilirsiniz.)

```
https://raw.githubusercontent.com/huzunluartemis/TurkishAdblockList/main/src/HostsList.txt
https://adaway.org/hosts.txt
https://hosts-file.net/ad_servers.txt
https://pgl.yoyo.org/adservers/serverlist.php?hostformat=hosts&showintro=0&mimetype=plaintext
https://raw.githubusercontent.com/StevenBlack/hosts/master/hosts
https://someonewhocares.org/hosts/hosts
https://s3.amazonaws.com/lists.disconnect.me/simple_malvertising.txt
```

- Ana menüye dönün, güncellemeleri denetleyip uygulayın. Cihazınızı yeniden başlatın.

### Eklenti destekleyen tarayıcılar // ücretsiz, cihaz kök-erişimli değilse

Kiwi Browser veya Mozilla Firefox gibi eklenti desteği olan bir tarayıcıda, eklentiler bölümünden ublock origin kurarak aynı işlemleri uygulayabilirsiniz.

## Eklememeniz gereken listeler (Yutulanlar)

Bu liste sağlayıcıyı kullanarak altta belirtilen listeleri de otomatik olarak kullanmış olursunuz. Endişe etmeyin, onlar da sürekli en güncel halinde olacaklar. Anlayacağınız bunları eklemenize gerek yok.

```
https://raw.githubusercontent.com/deathbybandaid/piholeparser/master/Subscribable-Lists/ParsedBlacklists/Turk-adlist.txt
https://gitlab.com/anarcho-copy/block-fake-pdf-sites/-/raw/master/hosts
https://raw.githubusercontent.com/biroloter/Mobile-Ad-Hosts/master/hosts
https://raw.githubusercontent.com/bkrucarci/turk-adlist/master/hosts
https://raw.githubusercontent.com/deathbybandaid/piholeparser/master/Subscribable-Lists/ParsedBlacklists/AakList.txt
https://raw.githubusercontent.com/abp-filters/abp-filters-anti-cv/master/turkish.txt
```

## Özel Durumlar

Burada belirtilen durumlar dosyalarda bulunmadığı için manuel olarak ayarlamalıdır. Şu kurallara göre uygulayınız:

- karalisteye eklemek için: "adaway > your" lists alanına girin. altta "blacklist" seçili olduğundan emin olduktan sonra "+ (uçan buton)" işaretine tıklayıp belirtilen kısmı yazın.
- beyazlisteye eklemek için: "adaway > your" lists alanına girin. altta "whitelist" seçili olduğundan emin olduktan sonra "+ (uçan buton)" işaretine tıklayıp belirtilen kısmı yazın.

## Rahatsız Siteyi Nasıl Bildireceğim?

Engellenmesini uygun gördüğünüz siteleri [Hatalar](https://github.com/huzunluartemis/TurkishAdblockList/issues) kısmından bildirin, listeye ekleyelim ki diğer insanlar bunlarla uğraşmasınlar. 

## Uyarı

Bu makaledeki uygulamaların gizlilik sözleşmelerini okuyunuz. Eğer ne yaptığınızı bilmiyorsanız bu işlemlerden uzak durun. Her cihazın yapısı farklıdır, oluşabilecek sorunlardan makale editörü sorumlu tutulamaz.

## Lisans

![](https://www.gnu.org/graphics/gplv3-127x51.png)

You can use, study share and improve it at your will. Specifically you can redistribute and/or modify it under the terms of the [GNU General Public License](https://www.gnu.org/licenses/gpl-3.0.html) as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.