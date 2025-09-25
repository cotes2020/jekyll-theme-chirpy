# Changelog

## [7.3.1](https://github.com/cotes2020/jekyll-theme-chirpy/compare/v7.3.0...v7.3.1) (2025-07-26)

### Bug Fixes

* escape special JSON characters in search results ([#2481](https://github.com/cotes2020/jekyll-theme-chirpy/issues/2481)) ([7615d72](https://github.com/cotes2020/jekyll-theme-chirpy/commit/7615d72e9300a1514ef2fc8ec941ab2974ba7eb4))

## [7.3.0](https://github.com/cotes2020/jekyll-theme-chirpy/compare/v7.2.4...v7.3.0) (2025-05-18)

### Features

* **i18n:** add Catalan Spanish locale translation ([#2349](https://github.com/cotes2020/jekyll-theme-chirpy/issues/2349)) ([167c98c](https://github.com/cotes2020/jekyll-theme-chirpy/commit/167c98c781d0607c90ede8fc73eb43dffeea6abd))
* **i18n:** add Dutch locale ([#2076](https://github.com/cotes2020/jekyll-theme-chirpy/issues/2076)) ([981ddba](https://github.com/cotes2020/jekyll-theme-chirpy/commit/981ddba30e57934f9056b8d468f0d17db131e1e8))
* **i18n:** add Japanese locale ([#2295](https://github.com/cotes2020/jekyll-theme-chirpy/issues/2295)) ([571c90f](https://github.com/cotes2020/jekyll-theme-chirpy/commit/571c90f13011eb91d0e1392218f3953060b920c9))
* **i18n:** add persian language ([#2238](https://github.com/cotes2020/jekyll-theme-chirpy/issues/2238)) ([7d4d35c](https://github.com/cotes2020/jekyll-theme-chirpy/commit/7d4d35cd10109e78d60fbb6b25a9b205f780ad63))

### Bug Fixes

* avoid `mathjax` loading failure on page refresh ([#2389](https://github.com/cotes2020/jekyll-theme-chirpy/issues/2389)) ([401e2af](https://github.com/cotes2020/jekyll-theme-chirpy/commit/401e2af0f8a173d8437e03027c7aff558e8c0bde))
* improve accuracy of moving `img` element classes ([#2399](https://github.com/cotes2020/jekyll-theme-chirpy/issues/2399)) ([d0f8f95](https://github.com/cotes2020/jekyll-theme-chirpy/commit/d0f8f9553e41536eb84ae2fdd3f3bc9d13f7ef8c))
* prevent the search bar from moving when focused ([#2336](https://github.com/cotes2020/jekyll-theme-chirpy/issues/2336)) ([f744929](https://github.com/cotes2020/jekyll-theme-chirpy/commit/f7449299e88c71da2104f0007f2db23a8fa798be))
* recognize global theme mode ([#2357](https://github.com/cotes2020/jekyll-theme-chirpy/issues/2357)) ([7708adb](https://github.com/cotes2020/jekyll-theme-chirpy/commit/7708adbf30e6dea51a84311b86bc224739f656f6))
* **search:** avoid missing spaces between paragraphs in search results ([#2199](https://github.com/cotes2020/jekyll-theme-chirpy/issues/2199)) ([0eb7efa](https://github.com/cotes2020/jekyll-theme-chirpy/commit/0eb7efa7f53508bf6b48eb9d773d5c5047c3c525))
* **ui:** fix incomplete border color on hover for tags ([#2359](https://github.com/cotes2020/jekyll-theme-chirpy/issues/2359)) ([c626447](https://github.com/cotes2020/jekyll-theme-chirpy/commit/c62644759cb4e0e07f7ee6eb9503ef69be62371b))

### Improvements

* **seo:** improve accessibility and aligns with best practices ([#2289](https://github.com/cotes2020/jekyll-theme-chirpy/issues/2289)) ([54d4d59](https://github.com/cotes2020/jekyll-theme-chirpy/commit/54d4d59d22ac543a14bfbd9bb3d6fb6756056041))

## [7.2.4](https://github.com/cotes2020/jekyll-theme-chirpy/compare/v7.2.3...v7.2.4) (2024-12-21)

### Bug Fixes

* toc not visible when switching from mobile to desktop mode ([#2139](https://github.com/cotes2020/jekyll-theme-chirpy/issues/2139)) ([32051da](https://github.com/cotes2020/jekyll-theme-chirpy/commit/32051dad03cb8f60fa4206969377b9674f9a3f0c))
* **ui:** left borderline of TOC is notched ([#2140](https://github.com/cotes2020/jekyll-theme-chirpy/issues/2140)) ([8a4d0bc](https://github.com/cotes2020/jekyll-theme-chirpy/commit/8a4d0bc4ee9e142a11401cad80bc9605878f121d))

## [7.2.3](https://github.com/cotes2020/jekyll-theme-chirpy/compare/v7.2.2...v7.2.3) (2024-12-15)

### Bug Fixes

* refreshing mermaid theme may fail ([#2113](https://github.com/cotes2020/jekyll-theme-chirpy/issues/2113)) ([2f00d41](https://github.com/cotes2020/jekyll-theme-chirpy/commit/2f00d41861f1b06c2ff7fa4e67e14e647c3c34b0))
* **ui:** gap between TOC entries is inconsistent ([#2119](https://github.com/cotes2020/jekyll-theme-chirpy/issues/2119)) ([1b4e318](https://github.com/cotes2020/jekyll-theme-chirpy/commit/1b4e318dc1cd57da812e11bf69ebb06083c213fc))
* **ui:** slow script loading hides TOC fade-up effect in desktop ([#2120](https://github.com/cotes2020/jekyll-theme-chirpy/issues/2120)) ([e0c3faf](https://github.com/cotes2020/jekyll-theme-chirpy/commit/e0c3fafa470eb12bd04ffdf198018bc28b6de20d))

## [7.2.2](https://github.com/cotes2020/jekyll-theme-chirpy/compare/v7.2.1...v7.2.2) (2024-12-06)

### Bug Fixes

* js files in subdirectories are excluded from the site output ([#2101](https://github.com/cotes2020/jekyll-theme-chirpy/issues/2101)) ([f55cc31](https://github.com/cotes2020/jekyll-theme-chirpy/commit/f55cc31dbd0e7455328c80c7ef38186ad8e54099))

## [7.2.1](https://github.com/cotes2020/jekyll-theme-chirpy/compare/v7.2.0...v7.2.1) (2024-12-05)

### Bug Fixes

* **build:** exclude `purgecss.js` from output files ([#2090](https://github.com/cotes2020/jekyll-theme-chirpy/issues/2090)) ([976e1a1](https://github.com/cotes2020/jekyll-theme-chirpy/commit/976e1a184b3dbe08991e8a50db4d5d7f8a0b7090))
* correct the import condition for theme script ([#2075](https://github.com/cotes2020/jekyll-theme-chirpy/issues/2075)) ([a16aa7d](https://github.com/cotes2020/jekyll-theme-chirpy/commit/a16aa7d41e3c3cb28649bfa1361e8bcb91b9ca47))
* ensure pageviews are fetched after DOM is loaded ([#2071](https://github.com/cotes2020/jekyll-theme-chirpy/issues/2071)) ([b4019f3](https://github.com/cotes2020/jekyll-theme-chirpy/commit/b4019f3517e4a3284df51567d29938cb12bf3acc))
* **toc:** resume fade up animation in desktop mode ([#2085](https://github.com/cotes2020/jekyll-theme-chirpy/issues/2085)) ([8280adb](https://github.com/cotes2020/jekyll-theme-chirpy/commit/8280adb901b9d15cc1bc18009553aae8746121d8))

## [7.2.0](https://github.com/cotes2020/jekyll-theme-chirpy/compare/v7.1.1...v7.2.0) (2024-11-28)

### Features

* show toc on mobile screens ([#1964](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1964)) ([8a064a5](https://github.com/cotes2020/jekyll-theme-chirpy/commit/8a064a5e5a95cd22aa654f7c80da09d107262508))
* support vertical scrolling for toc in desktop mode ([#2064](https://github.com/cotes2020/jekyll-theme-chirpy/issues/2064)) ([5265b03](https://github.com/cotes2020/jekyll-theme-chirpy/commit/5265b039741555943f9a6f0451287aefb6810f28))

### Bug Fixes

* pagination error when pinned posts exceed the page size ([#1965](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1965)) ([93f616b](https://github.com/cotes2020/jekyll-theme-chirpy/commit/93f616b25d7ed6c4f090c50c8663f8c1f59947f4))

### Improvements

* modular sass architecture ([#2052](https://github.com/cotes2020/jekyll-theme-chirpy/issues/2052)) ([35c794c](https://github.com/cotes2020/jekyll-theme-chirpy/commit/35c794cf5896565430389f35c660b88a93cebb17))
* speed up page rendering and jekyll build process ([#2034](https://github.com/cotes2020/jekyll-theme-chirpy/issues/2034)) ([65f960c](https://github.com/cotes2020/jekyll-theme-chirpy/commit/65f960c31a734b5306a8b919040c3aae9b783efd))

## [7.1.1](https://github.com/cotes2020/jekyll-theme-chirpy/compare/v7.1.0...v7.1.1) (2024-09-23)

### Bug Fixes

* **i18n:** correct fr-FR translations ([#1949](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1949)) ([367262e](https://github.com/cotes2020/jekyll-theme-chirpy/commit/367262e74d1005bddf1328bb2b3a2b9e152c0086))
* **pwa:** site baseurl not passed to `app.js` ([#1955](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1955)) ([5a63244](https://github.com/cotes2020/jekyll-theme-chirpy/commit/5a63244721d21b1ad3a0ae83420723a2f0379e8b))

## [7.1.0](https://github.com/cotes2020/jekyll-theme-chirpy/compare/v7.0.1...v7.1.0) (2024-08-27)

### Features

* add Bluesky social links ([#1759](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1759)) ([0102aba](https://github.com/cotes2020/jekyll-theme-chirpy/commit/0102abae062be24ec289fb7facb11950aca79e3f))
* add Reddit social option ([#1836](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1836)) ([8673e13](https://github.com/cotes2020/jekyll-theme-chirpy/commit/8673e1335f0771eac364d0a2866f27476d61a58b))
* add Threads social links ([#1837](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1837)) ([e3a78b6](https://github.com/cotes2020/jekyll-theme-chirpy/commit/e3a78b6243f7056105d72185bb6e94b436834e5b))
* **analytics:** add fathom analytics ([#1913](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1913)) ([befc4ce](https://github.com/cotes2020/jekyll-theme-chirpy/commit/befc4ce9c5026f67f99bce66e223d056229f0bdb))
* **dev:** add vscode tasks ([#1843](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1843)) ([e4db1a1](https://github.com/cotes2020/jekyll-theme-chirpy/commit/e4db1a176f3f69f676cbc0bf27b1d5a657ece05e))
* **dev:** support vscode dev-container ([#1781](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1781)) ([1e3d4a6](https://github.com/cotes2020/jekyll-theme-chirpy/commit/1e3d4a6323ba3eed06a57f8bf1b2edefd890b127))
* **ui:** improve block quote layout ([80bd792](https://github.com/cotes2020/jekyll-theme-chirpy/commit/80bd7928a02c75c843a550bd377d11b574e8bfda))
* **ui:** improve visibility of inline code ([#1831](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1831)) ([c876731](https://github.com/cotes2020/jekyll-theme-chirpy/commit/c876731901784a72ef9d2e9e2936df65ddff5f61))
* **ui:** make `info-prompt` icon looks like the letter "i" ([#1835](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1835)) ([a07a57e](https://github.com/cotes2020/jekyll-theme-chirpy/commit/a07a57ec922249d3a22da56fbcb30d83eadef728))
* **ui:** set `<kbd>` font to 'Lato' ([64c7262](https://github.com/cotes2020/jekyll-theme-chirpy/commit/64c7262245e878534971a2e3a2630b614daf72f3))

### Bug Fixes

* adapt the giscus localization parameter ([#1810](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1810)) ([0709854](https://github.com/cotes2020/jekyll-theme-chirpy/commit/0709854dc8f6099d38c2578967a02f73b4be0dc8))
* avoid caching pageviews data ([#1849](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1849)) ([979f86c](https://github.com/cotes2020/jekyll-theme-chirpy/commit/979f86cf64e1fcace4231fb070c7e6398fd4e5ec))
* remove extra dual-mode images from lightbox ([#1883](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1883)) ([5c5910f](https://github.com/cotes2020/jekyll-theme-chirpy/commit/5c5910f1fc661ec3dce203ac961c7f64f1991895))

## [7.0.1](https://github.com/cotes2020/jekyll-theme-chirpy/compare/v7.0.0...v7.0.1) (2024-05-18)

### Bug Fixes

* **analytics:** goatcounter pv greater than 1K cannot be converted to numbers ([#1762](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1762)) ([33a1fa7](https://github.com/cotes2020/jekyll-theme-chirpy/commit/33a1fa7cae2181625e2f335708d59de4dd20ee7d))
* audio/video path apply variable `media_subpath` ([#1745](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1745)) ([00a27a1](https://github.com/cotes2020/jekyll-theme-chirpy/commit/00a27a1b85f665d0642b77babd54c6903fbdeb22))

## [7.0.0](https://github.com/cotes2020/jekyll-theme-chirpy/compare/v6.5.5...v7.0.0) (2024-05-11)

### ⚠ BREAKING CHANGES

* optimize the resource hints (#1717)
* rename media-url file and related parameters (#1651)
* rename comment setting parameter (#1563)
* **analytics:** add post pageviews for GoatCounter (#1543)

### Features

* add cloudflare web analytics ([#1723](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1723)) ([c17fba4](https://github.com/cotes2020/jekyll-theme-chirpy/commit/c17fba44f53767c9dfaa8d92cfc6e275e5977f8a))
* add support for embed video files ([#1558](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1558)) ([9592146](https://github.com/cotes2020/jekyll-theme-chirpy/commit/9592146ca392236e69ee358412ecc32ef1662127))
* add support for giscus strict title matching ([#1614](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1614)) ([700fd5b](https://github.com/cotes2020/jekyll-theme-chirpy/commit/700fd5bad7272dd950f861e8550215cd8fafb413))
* **analytics:** add post pageviews for GoatCounter ([#1543](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1543)) ([b641b3f](https://github.com/cotes2020/jekyll-theme-chirpy/commit/b641b3f1f2e54bcfe96d8dff46d4f94186492d98))
* **analytics:** add Umami and Matomo tracking codes ([#1658](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1658)) ([61bdca2](https://github.com/cotes2020/jekyll-theme-chirpy/commit/61bdca2db45179cd0d1b4b885a4c4864e3ffa3c1))
* change site verification settings ([#1561](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1561)) ([e436387](https://github.com/cotes2020/jekyll-theme-chirpy/commit/e4363871b5be0608d2b92b8aff482825a8044c1b))
* **deps:** move `MathJax` configuration to a separate file ([#1670](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1670)) ([44f552c](https://github.com/cotes2020/jekyll-theme-chirpy/commit/44f552cbcee83d037de0e59496bf6bb19eea2691))
* display theme version in footer ([#1611](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1611)) ([8349314](https://github.com/cotes2020/jekyll-theme-chirpy/commit/834931486dc3e5ed544ce4ff47cd1b2bc45f42fd))
* **i18n:** allow `page.lang` to override `site.lang` ([#1586](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1586)) ([547b95c](https://github.com/cotes2020/jekyll-theme-chirpy/commit/547b95cc7ae35018dadcc01b6eb1dc8c8943e67e))
* make post description customizable ([#1602](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1602)) ([f865336](https://github.com/cotes2020/jekyll-theme-chirpy/commit/f865336c896e0db34edf8482a53e0e5d8f07ff95))
* **media:** support audio and video tag with multi sources ([#1618](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1618)) ([23be416](https://github.com/cotes2020/jekyll-theme-chirpy/commit/23be4162b3f8598db14dc5b39726932ccf2cdc23))

### Bug Fixes

* make TOC title and entries visible at the same time ([#1711](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1711)) ([e0950fc](https://github.com/cotes2020/jekyll-theme-chirpy/commit/e0950fc973d029dc65d0bc1bd68f3d11242527c8))
* mode toggle not outlined when receiving keyboard focus ([#1690](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1690)) ([cd37f63](https://github.com/cotes2020/jekyll-theme-chirpy/commit/cd37f63a0144e0499ea991d3309da064ad5eccea))
* prevent footnote back arrow from becoming an emoji ([#1716](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1716)) ([8608147](https://github.com/cotes2020/jekyll-theme-chirpy/commit/8608147fb5037804695d93496c62f96b9c41e9cd))
* **pwa:** skip range requests in service worker ([#1672](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1672)) ([76d58fe](https://github.com/cotes2020/jekyll-theme-chirpy/commit/76d58fe0ffdc4bd1df35b60815e97560c3564700))
* search result prompt is empty ([#1583](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1583)) ([8a2afae](https://github.com/cotes2020/jekyll-theme-chirpy/commit/8a2afae6cab8fc9639be0a866b71699c8a80084c))
* use `https` for Weibo sharing URL ([#1612](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1612)) ([8e5fbb7](https://github.com/cotes2020/jekyll-theme-chirpy/commit/8e5fbb7a74d04a4b3cdde69bcc821f8ccd1a3bc0))

### Improvements

* improve \<hr> visibility in dark mode ([#1565](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1565)) ([4ddd5c4](https://github.com/cotes2020/jekyll-theme-chirpy/commit/4ddd5c437046a1e70cf396113e2351c452a25493))
* lean bootstrap javascript ([#1734](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1734)) ([ddb48ed](https://github.com/cotes2020/jekyll-theme-chirpy/commit/ddb48eda52827aae16aff720212d7b6d2d8647f9))
* rename comment setting parameter ([#1563](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1563)) ([f8390d4](https://github.com/cotes2020/jekyll-theme-chirpy/commit/f8390d4384600fb015728b1b186570fa58ca216f))
* replace jQuery with Vanilla JS ([#1681](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1681)) ([fe7afa3](https://github.com/cotes2020/jekyll-theme-chirpy/commit/fe7afa379f0af0ca98a207f85bdc0fa98575b1ad))
* simplify mode toggle script ([#1692](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1692)) ([d4a6d64](https://github.com/cotes2020/jekyll-theme-chirpy/commit/d4a6d640bd6d4ab185faf96c0255369a9903ee1d))
* tree shaking Bootstrap CSS ([#1736](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1736)) ([363a3d9](https://github.com/cotes2020/jekyll-theme-chirpy/commit/363a3d936bbd688fa4f28527e85ef7dd3fe3a79b))

### Changes

* optimize the resource hints ([#1717](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1717)) ([dcb0add](https://github.com/cotes2020/jekyll-theme-chirpy/commit/dcb0add47bf1adf92215514f1ccfa4661d5215be))
* rename media-url file and related parameters ([#1651](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1651)) ([9f8aeaa](https://github.com/cotes2020/jekyll-theme-chirpy/commit/9f8aeaadbfef9967a9b0a9dd323d8bed46e14d9f))

## [6.5.5](https://github.com/cotes2020/jekyll-theme-chirpy/compare/v6.5.4...v6.5.5) (2024-03-23)

### Bug Fixes

* **post:** correct the image URLs ([#1627](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1627)) ([2d649aa](https://github.com/cotes2020/jekyll-theme-chirpy/commit/2d649aae0e40a24db1ab0d46fa474294e96cb135))

## [6.5.4](https://github.com/cotes2020/jekyll-theme-chirpy/compare/v6.5.3...v6.5.4) (2024-03-22)

### Bug Fixes

* correct the attribute for the Twitter social image ([#1615](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1615)) ([cfe44f2](https://github.com/cotes2020/jekyll-theme-chirpy/commit/cfe44f204bcec8e05f498512ec50878e626a124f))
* **seo:** correct social preview image path inside `<meta>` tag ([#1623](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1623)) ([74cf57a](https://github.com/cotes2020/jekyll-theme-chirpy/commit/74cf57aaacf6674057e6f33240a22f4888cfe88f))

## [6.5.3](https://github.com/cotes2020/jekyll-theme-chirpy/compare/v6.5.2...v6.5.3) (2024-03-07)

### Changes

* replace `polyfill.io` with `cdnjs` hosted link ([#1598](https://github.com/cotes2020/jekyll-theme-chirpy/pull/1598)) ([75a3d73](https://github.com/cotes2020/jekyll-theme-chirpy/commit/75a3d7399b257256a09d602cbe01062fe1cdf68d))

## [6.5.2](https://github.com/cotes2020/jekyll-theme-chirpy/compare/v6.5.1...v6.5.2) (2024-02-29)

### Bug Fixes

* correct the base URL parameter name ([#1576](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1576)) ([19d6baf](https://github.com/cotes2020/jekyll-theme-chirpy/commit/19d6bafbe1a60614e0d63b961bc73c342a9f6f33)), closes [#1553](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1553)

## [6.5.1](https://github.com/cotes2020/jekyll-theme-chirpy/compare/v6.5.0...v6.5.1) (2024-02-26)

### Bug Fixes

* correct the generation of relative resource paths ([#1553](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1553)) ([89b9625](https://github.com/cotes2020/jekyll-theme-chirpy/commit/89b962557a56ccc13eba3c9c20b4270ee9d30042))

## [6.5.0](https://github.com/cotes2020/jekyll-theme-chirpy/compare/v6.4.2...v6.5.0) (2024-02-14)

### Features

* add `pwa.cache.*` option to precisely control caching ([#1501](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1501)) ([1127c43](https://github.com/cotes2020/jekyll-theme-chirpy/commit/1127c43823aac4db7fd80d5bb706ae7b1e129dc6))
* add analytics support for GoatCounter ([#1526](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1526)) ([90693ff](https://github.com/cotes2020/jekyll-theme-chirpy/commit/90693ff95e72ca4b5135a7b454a6ab521b995b3e))

### Bug Fixes

* correct the Twitter Card in social share preview ([#1498](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1498)) ([74f1662](https://github.com/cotes2020/jekyll-theme-chirpy/commit/74f16623c9c4877ef36ac52e8b69c19d1d9a82ba))
* missing "/" at the end of URLs for categories and tags in breadcrumb ([#1495](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1495)) ([02e296e](https://github.com/cotes2020/jekyll-theme-chirpy/commit/02e296ed75b7906b2d112c67f9054f5d71919de9))

### Improvements

* allow no social links to be configured ([#1494](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1494)) ([4facf5b](https://github.com/cotes2020/jekyll-theme-chirpy/commit/4facf5b390eeba612ca439f3354c5d2d881aac56))
* allow TOC to start at heading 3 ([#1512](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1512)) ([bbbb66b](https://github.com/cotes2020/jekyll-theme-chirpy/commit/bbbb66b489a3bf2b878947336fe894e8ea2ae3f5))
* enable equation numbering in MathJax ([#1520](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1520)) ([c13ec31](https://github.com/cotes2020/jekyll-theme-chirpy/commit/c13ec311636d5e057c6895e353e1c1a4e570f582))

## [6.4.2](https://github.com/cotes2020/jekyll-theme-chirpy/compare/v6.4.1...v6.4.2) (2024-01-13)

### Bug Fixes

* resume the `blockquote` display type ([#1480](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1480)) ([c7cfde0](https://github.com/cotes2020/jekyll-theme-chirpy/commit/c7cfde093020c73ca9a1b83437eb600379e05918)), closes [#1449](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1449)

## [6.4.1](https://github.com/cotes2020/jekyll-theme-chirpy/compare/v6.4.0...v6.4.1) (2024-01-10)

### Bug Fixes

* `og:image` URL is incorrect ([#1468](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1468)) ([b2d1cb6](https://github.com/cotes2020/jekyll-theme-chirpy/commit/b2d1cb68db659270aac537d2aa8d4b806fa6991d)), closes [#1463](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1463)

## [6.4.0](https://github.com/cotes2020/jekyll-theme-chirpy/compare/v6.3.1...v6.4.0) (2024-01-10)

### Features

* add bilibili embed video support ([#1406](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1406)) ([4a2b89d](https://github.com/cotes2020/jekyll-theme-chirpy/commit/4a2b89d0b698d672486349131a89025fa47afcb6))
* add site-wide social preview image settings ([#1463](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1463)) ([241bb4d](https://github.com/cotes2020/jekyll-theme-chirpy/commit/241bb4df7878cff7f82014df660874a1dcddba76))

### Bug Fixes

*  image float breaks quotes and prompts ([#1449](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1449)) ([ea2d238](https://github.com/cotes2020/jekyll-theme-chirpy/commit/ea2d238bd8adc018256862e05a5092311c87a671)), closes [#1441](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1441)
* url-less authors should not have empty links ([#1410](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1410)) ([2a4fbf6](https://github.com/cotes2020/jekyll-theme-chirpy/commit/2a4fbf6a7925da610a75c498116da7cf9ba857d7)), closes [#1403](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1403)

### Improvements

* include the latest posts in the "Recently Updated" list ([#1456](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1456)) ([82d8f2d](https://github.com/cotes2020/jekyll-theme-chirpy/commit/82d8f2db984711f334f55b6af5098ec16770e824))

## [6.3.1](https://github.com/cotes2020/jekyll-theme-chirpy/compare/v6.3.0...v6.3.1) (2023-11-12)

### Bug Fixes

* **home:** responsive gap at the bottom of preview image ([1a977a8](https://github.com/cotes2020/jekyll-theme-chirpy/commit/1a977a87a0da1cff35d0896cf9265c31034841a6))

## [6.3.0](https://github.com/cotes2020/jekyll-theme-chirpy/compare/v6.2.3...v6.3.0) (2023-11-10)

### Features

* add Mastodon sharing link ([#1344](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1344)) ([2bf87e0](https://github.com/cotes2020/jekyll-theme-chirpy/commit/2bf87e0de7928f325811e1bb96cfcaefdf6cf66a)), closes [#1324](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1324)

### Bug Fixes

* **home:** crop the preview image that doesn't match 1.91:1 ([#1325](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1325)) ([5810bcd](https://github.com/cotes2020/jekyll-theme-chirpy/commit/5810bcd1d7b83e111017831fa82c368a6b15c7cd))
* resume lazy loading for Twitch videos ([#1326](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1326)) ([9f174d9](https://github.com/cotes2020/jekyll-theme-chirpy/commit/9f174d9088e5c83a5e0c4630336cea65e199c553)), closes [#1267](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1267)

## [6.2.3](https://github.com/cotes2020/jekyll-theme-chirpy/compare/v6.2.2...v6.2.3) (2023-10-10)

### Bug Fixes

* avoid `utterances` initialization failure  ([#1234](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1234)) ([b34661e](https://github.com/cotes2020/jekyll-theme-chirpy/commit/b34661efd72e8697fd5b30ba7e55c86c7dd10338))
* **home:** avoid LQIP dirty data passing to the next post ([#1278](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1278)) ([109725d](https://github.com/cotes2020/jekyll-theme-chirpy/commit/109725d2dc56e329c60a876e9ce4094513fd36a5))
* **posts:** code snippet clipboard tooltip missing title ([#1246](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1246)) ([726085c](https://github.com/cotes2020/jekyll-theme-chirpy/commit/726085c6478e7a9dc2cc57189b2dcbc85d90f048))
* **posts:** resume target highlighting for superscripts and footnotes ([#1253](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1253)) ([0465a98](https://github.com/cotes2020/jekyll-theme-chirpy/commit/0465a985dc5262fa2043540f4eddafa251f917a3))

### Improvements

* **comments:** lazy load `giscus` ([#1254](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1254)) ([e9c9206](https://github.com/cotes2020/jekyll-theme-chirpy/commit/e9c920641b9c97594fa078ea89747d77eb7e7493))
* **core:** replace `lazysizes` with browser-level lazy loading ([#1267](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1267)) ([bf3a34d](https://github.com/cotes2020/jekyll-theme-chirpy/commit/bf3a34d0544b49fcf40f57080c4d6b4ff44750c4))
* **layout:** improve margins for tail block ([#1243](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1243)) ([13a3c3c](https://github.com/cotes2020/jekyll-theme-chirpy/commit/13a3c3c906bb6c5a38314ea27b6cf3767df94b27))
* **layout:** optimize the main block height calculation ([#1249](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1249)) ([73e171b](https://github.com/cotes2020/jekyll-theme-chirpy/commit/73e171b0fbce4a542e2141d7e2b1144450571ce1))
* **pwa:** enhance cache privacy protection ([#1275](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1275)) ([2d56597](https://github.com/cotes2020/jekyll-theme-chirpy/commit/2d56597571aaafa92251d192861ea69cce3e83d2))
* **ui:** standardize metadata text styles ([#1295](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1295)) ([2574118](https://github.com/cotes2020/jekyll-theme-chirpy/commit/2574118f40a956184705f87dea4d88e7c246a055))
* **ux:** render background color before loading preview image ([#1298](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1298)) ([42bf39e](https://github.com/cotes2020/jekyll-theme-chirpy/commit/42bf39e21c9a497aecc3e9b4549e2fc3ba4a1e4a))

## [6.2.2](https://github.com/cotes2020/jekyll-theme-chirpy/compare/v6.2.1...v6.2.2) (2023-09-10)

### Bug Fixes

* **sidebar:** contact icons not stacking ([#1224](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1224)) ([273b389](https://github.com/cotes2020/jekyll-theme-chirpy/commit/273b389c512f13693ed6cdf57d256ac21deae97c))

## [6.2.1](https://github.com/cotes2020/jekyll-theme-chirpy/compare/v6.2.0...v6.2.1) (2023-09-10)

### Bug Fixes

* **pwa:** installation failure caused by outdated cache entries ([4da7406](https://github.com/cotes2020/jekyll-theme-chirpy/commit/4da7406dfea112a4a2b1db5615ecf2672be6694f))

## [6.2.0](https://github.com/cotes2020/jekyll-theme-chirpy/compare/v6.1.0...v6.2.0) (2023-09-10)

### Features

* **layout:** center the footer ([41b8f9f](https://github.com/cotes2020/jekyll-theme-chirpy/commit/41b8f9f519e5f5f69e9a123b38b06bade2271a82))
* **posts:** render heading 4 in TOC ([#1023](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1023)) ([229c2a2](https://github.com/cotes2020/jekyll-theme-chirpy/commit/229c2a2e2b109fc2eca85be548f1dd97234e44c4))
* **ui:** redesign the pagination button on home page ([62bcd60](https://github.com/cotes2020/jekyll-theme-chirpy/commit/62bcd601fcadc602c81672b1d4b937231396c3c0))
* **ui:** update the twitter icon ([#1221](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1221)) ([aff7566](https://github.com/cotes2020/jekyll-theme-chirpy/commit/aff75667749769644f990d3dc9b0720c7d96d14d))

### Improvements

* **core:** speed up the Jekyll build times ([#1163](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1163)) ([0d4103d](https://github.com/cotes2020/jekyll-theme-chirpy/commit/0d4103d47bc9cff93918bb09a2957737cc3c9fe0))
* refactor using semantic HTML ([#1207](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1207)) ([505e314](https://github.com/cotes2020/jekyll-theme-chirpy/commit/505e314a3142c332e39365fbe2dac23df1bf0abe)), closes [#1196](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1196)
* **ui:** improve code snippet design ([4f86b04](https://github.com/cotes2020/jekyll-theme-chirpy/commit/4f86b04a8487ebbf4a6d0d70b0c3ece79e9269f3))
* **ui:** improve web accessibility ([#447](https://github.com/cotes2020/jekyll-theme-chirpy/issues/447)) ([37c9764](https://github.com/cotes2020/jekyll-theme-chirpy/commit/37c976499ead51c1d88e8e8213366240a72adebc))

## [6.1.0](https://github.com/cotes2020/jekyll-theme-chirpy/compare/v6.0.0...v6.1.0) (2023-07-02)

### Features

* **i18n:** add Thai locale file ([#1087](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1087)) ([a60e907](https://github.com/cotes2020/jekyll-theme-chirpy/commit/a60e90791d24811caff78e21c71dc85d6a729438))

### Bug Fixes

* missing xml escape for `alt` of preview image ([#1113](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1113)) ([8b0fbf5](https://github.com/cotes2020/jekyll-theme-chirpy/commit/8b0fbf5a834276f273274e4d614edd71e339cbb0))
* the cached image is covered by shimmer ([#1100](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1100)) ([df8ff54](https://github.com/cotes2020/jekyll-theme-chirpy/commit/df8ff546ec1c8d21a3d25e0124665001fcf756f3))
* **ui:** min-height of `page` layout exceeds the mobile screen ([73af591](https://github.com/cotes2020/jekyll-theme-chirpy/commit/73af59194ab935d38b89d298fea0e96e13be7cb7))
* **webfont:** resume semi-bold of font family `Source Sans Pro` ([c4da99c](https://github.com/cotes2020/jekyll-theme-chirpy/commit/c4da99c7ea5d6e32b1f1b815d7d8d6ae7b0f55de))

### Improvements

* **build:** use `jekyll-include-cache` plugin to reduce build time ([#1098](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1098)) ([4fe145e](https://github.com/cotes2020/jekyll-theme-chirpy/commit/4fe145e9809ee1b370d9891135939534751462d0)), closes [#1094](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1094)
* CJK characters of the "Search Cancel" button will wrap ([#1105](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1105)) ([b6d1992](https://github.com/cotes2020/jekyll-theme-chirpy/commit/b6d1992f85ec543220e826087dcc89870e7e2c00))
* **ui:** avoid blank space at the bottom of the homepage preview image ([ce2f6f5](https://github.com/cotes2020/jekyll-theme-chirpy/commit/ce2f6f5abef7a8b874e08d1f18c1fd002650dbf1))
* **ui:** improve hover color of sidebar nav items in light mode ([728094d](https://github.com/cotes2020/jekyll-theme-chirpy/commit/728094d1ba67a1e7c0a11e1c6c69bf87af9a767b))

## [6.0.1](https://github.com/cotes2020/jekyll-theme-chirpy/compare/v6.0.0...v6.0.1) (2023-05-19)

### Bug Fixes

* **home:** preview image missing `[alt]` and `img_path` ([#1044](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1044)) ([aba9468](https://github.com/cotes2020/jekyll-theme-chirpy/commit/aba9468b5332802db961166889d4c4a84e404a2c))
* **layout:** restore the margin bottom of the main area ([#1047](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1047)) ([eb40f51](https://github.com/cotes2020/jekyll-theme-chirpy/commit/eb40f51c84b011a7c301279527f544ad27efd5eb))
* **post, page:** image link loses shimmer effect ([#1046](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1046)) ([3bd881d](https://github.com/cotes2020/jekyll-theme-chirpy/commit/3bd881da70d685d10659f47bfe0e79cd02e7af92))
* **typography:** long string for update-list is not truncated ([#1050](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1050)) ([a51d31c](https://github.com/cotes2020/jekyll-theme-chirpy/commit/a51d31c55a37fbe034f0b0f699f4df0b6a14ba8f)), closes [#1049](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1049)

## [6.0.0](https://github.com/cotes2020/jekyll-theme-chirpy/compare/v5.6.1...v6.0.0) (2023-05-16)

### ⚠ BREAKING CHANGES

* rename assets origin configuration files

### Features

* add a hook to insert custom metadata in `head` tag ([#1015](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1015)) ([fe20341](https://github.com/cotes2020/jekyll-theme-chirpy/commit/fe203417d993508eedf5b9044fe53c4a566e44f9))
* **i18n:** add sl-SI.yml with slovenian translations ([#989](https://github.com/cotes2020/jekyll-theme-chirpy/issues/989)) ([42a700a](https://github.com/cotes2020/jekyll-theme-chirpy/commit/42a700aa37889faa32d7ec1f6776ce4b9d845dc4))
* **i18n:** add Traditional Chinese (Taiwan) localization file ([#961](https://github.com/cotes2020/jekyll-theme-chirpy/issues/961)) ([d97f95f](https://github.com/cotes2020/jekyll-theme-chirpy/commit/d97f95fca0bcd450ea50709ffba0217f7e65d339))
* **i18n:** added Swedish localization file ([#969](https://github.com/cotes2020/jekyll-theme-chirpy/issues/969)) ([fe70479](https://github.com/cotes2020/jekyll-theme-chirpy/commit/fe7047959e3694c6e603e764ded30dacd49e6aa9))
* support hiding the modification date of a post ([#1020](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1020)) ([8da583d](https://github.com/cotes2020/jekyll-theme-chirpy/commit/8da583d403456f6460ec1a6ebcbb0c2ca8127ff6))
* **ui:** improve code snippet design ([6d99f5c](https://github.com/cotes2020/jekyll-theme-chirpy/commit/6d99f5cc36a69e5ccff51f81ba448c798d92e12e))
* **ui:** improve the design for top bar ([83f1c34](https://github.com/cotes2020/jekyll-theme-chirpy/commit/83f1c34f92d85f3953ca9c9818be5399962bf1c9))
* **ui:** new design footer content layout ([3210c59](https://github.com/cotes2020/jekyll-theme-chirpy/commit/3210c59466150dc04b4e4bdfc1ffd0e38adcff43))
* **ui:** redesign the sidebar ([83bbe4a](https://github.com/cotes2020/jekyll-theme-chirpy/commit/83bbe4ac939edfd1706e68c080562e3462f83519))
* **ui:** show preview image in home page ([97b8dfe](https://github.com/cotes2020/jekyll-theme-chirpy/commit/97b8dfeed6ce7677f6472e28dc3b03f3c2968b12))

### Bug Fixes

* parameter parsing error in image URL ([#1022](https://github.com/cotes2020/jekyll-theme-chirpy/issues/1022)) ([ee88cec](https://github.com/cotes2020/jekyll-theme-chirpy/commit/ee88cec270ea5938f98913a3edf28a684cfbd6c0))
* **rss:** double quotes in the post title will break the XML structure ([#965](https://github.com/cotes2020/jekyll-theme-chirpy/issues/965)) ([1719d81](https://github.com/cotes2020/jekyll-theme-chirpy/commit/1719d81d00b32b107c35b3903089be84a9b28a6c))

### refactor

* rename assets origin configuration files ([c283e77](https://github.com/cotes2020/jekyll-theme-chirpy/commit/c283e7782fa9562d82d9855fd280a573fd58c75f))

### Improvements

* **assets:** reduce HTTP requests to CDN ([9d97120](https://github.com/cotes2020/jekyll-theme-chirpy/commit/9d971201978e993a9af337d9cd5396a1ea225f00))
* calculate heading font size dynamically ([#983](https://github.com/cotes2020/jekyll-theme-chirpy/issues/983)) ([52f5ee9](https://github.com/cotes2020/jekyll-theme-chirpy/commit/52f5ee9cd3f92a6e8f25eaa203831546cda85db6))
* **i18n:** set the global default locales to "en" ([#979](https://github.com/cotes2020/jekyll-theme-chirpy/issues/979)) ([61fdbcb](https://github.com/cotes2020/jekyll-theme-chirpy/commit/61fdbcb83a3601ecae62ec230602b94a5eb832e1))
* **tools:** avoid initialization interruption in single branch forks ([#992](https://github.com/cotes2020/jekyll-theme-chirpy/issues/992)) ([e90461a](https://github.com/cotes2020/jekyll-theme-chirpy/commit/e90461aa3c81633863db6a12c5924ddba33bd08e))
* **ui:** improve categories color in dark mode ([414dd13](https://github.com/cotes2020/jekyll-theme-chirpy/commit/414dd132aed70f4bd96cb712d00eacc82d2753e9))
* **ui:** improve hover effect for post preview cards ([7626e4d](https://github.com/cotes2020/jekyll-theme-chirpy/commit/7626e4d00544346a46b6e5ff2f3a99d234defe09))
* **ui:** improve hover effect of trending tags ([34499f0](https://github.com/cotes2020/jekyll-theme-chirpy/commit/34499f0c927ce8fea3705dc2f0f0e6805cabda43))
* **ui:** improve inline code in light mode ([e38309f](https://github.com/cotes2020/jekyll-theme-chirpy/commit/e38309f3bd1302ffe60b682136b6efaf96f4d9ae))
* **ui:** improve related posts design ([2918da9](https://github.com/cotes2020/jekyll-theme-chirpy/commit/2918da9f29465618d557c082ff3a2f23d7519049))
* **ui:** improve the color of prompts in dark mode ([8cbbcfa](https://github.com/cotes2020/jekyll-theme-chirpy/commit/8cbbcfa26da0addd88affada23a65770250f2404))
* **ui:** lighten the link color in light-mode ([7c23a4e](https://github.com/cotes2020/jekyll-theme-chirpy/commit/7c23a4ebc53b9e231c214e04f8ac0803cbcdb720))
* **ui:** mute the marker in lists ([0c80552](https://github.com/cotes2020/jekyll-theme-chirpy/commit/0c80552d772b874e2a161f1270294faa3af18d4a))
* **ui:** uniform the muted text color ([aadf939](https://github.com/cotes2020/jekyll-theme-chirpy/commit/aadf9393d5c7f7528d453c4e68eba4f5cbb85bd9))
* **ux:** improve LQIP fade in effect ([003e7b6](https://github.com/cotes2020/jekyll-theme-chirpy/commit/003e7b60c93988a7bfae4c03a8346d4f8a5f0bb6))

## [5.6.1](https://github.com/cotes2020/jekyll-theme-chirpy/compare/v5.6.0...v5.6.1) (2023-03-30)

### Bug Fixes

* **deps:** `tocbot` has no initialization detection ([#957](https://github.com/cotes2020/jekyll-theme-chirpy/issues/957)) ([8225174](https://github.com/cotes2020/jekyll-theme-chirpy/commit/8225174cb5e02fda7b3cc548ec821c876b0a5139))
* mode-toggle leads to Disqus loading failure ([#945](https://github.com/cotes2020/jekyll-theme-chirpy/issues/945)) ([6fec411](https://github.com/cotes2020/jekyll-theme-chirpy/commit/6fec411c18ca5689c467c7b216ddeda02df23623))
* pageviews not updated immediately ([8b4f99c](https://github.com/cotes2020/jekyll-theme-chirpy/commit/8b4f99c87f9a9227f47e84fb39d7b0f551d6f4dd))

## [5.6.0](https://github.com/cotes2020/jekyll-theme-chirpy/compare/v5.5.2...v5.6.0) (2023-03-17)

### Features

* change TOC plugin to `tocbot` ([#774](https://github.com/cotes2020/jekyll-theme-chirpy/issues/774)) ([02b7bd5](https://github.com/cotes2020/jekyll-theme-chirpy/commit/02b7bd5095a2affe5b4c5ed7b5b182baaf642ff3))
* **i18n:** add Greek Language Support. ([#903](https://github.com/cotes2020/jekyll-theme-chirpy/issues/903)) ([712a9b2](https://github.com/cotes2020/jekyll-theme-chirpy/commit/712a9b22401ce591cf4c0bb03fbdd1693fee30bb))
* **ux:** turn home page posts into clickable cards ([#895](https://github.com/cotes2020/jekyll-theme-chirpy/issues/895)) ([b85f633](https://github.com/cotes2020/jekyll-theme-chirpy/commit/b85f6330dea666350631c4461b742cdb54c5f052))

### Bug Fixes

* css selector string escaping vulnerability ([#888](https://github.com/cotes2020/jekyll-theme-chirpy/issues/888)) ([5c6ec9d](https://github.com/cotes2020/jekyll-theme-chirpy/commit/5c6ec9d06b6571e2c0efe6652078442dca8af477))
* mathematics cannot scroll horizontally ([#760](https://github.com/cotes2020/jekyll-theme-chirpy/issues/760)) ([4681df7](https://github.com/cotes2020/jekyll-theme-chirpy/commit/4681df715118a37ae1e91b588de0adb67f4e331a))
* notch status bar doesn't match theme color ([#918](https://github.com/cotes2020/jekyll-theme-chirpy/issues/918)) ([820ba62](https://github.com/cotes2020/jekyll-theme-chirpy/commit/820ba62e9e939090523a7077d01d01bd78ec84eb))
* some console snippets will be incompletely copied ([e8e4901](https://github.com/cotes2020/jekyll-theme-chirpy/commit/e8e4901e340dd7e5fc5f656dd3c7bcd6c97b886a))

## [5.5.2](https://github.com/cotes2020/jekyll-theme-chirpy/compare/v5.5.1...v5.5.2) (2023-01-30)

### Bug Fixes

* position of prompt icon is incorrect in paragraph on mobile ([5df953f](https://github.com/cotes2020/jekyll-theme-chirpy/commit/5df953f6c877e2aa3f1f4981c97a0b8007abe6d4))

## [5.5.1](https://github.com/cotes2020/jekyll-theme-chirpy/compare/v5.5.0...v5.5.1) (2023-01-29)

### Bug Fixes

* the icon position of the prompts in the list is incorrect ([0c9558d](https://github.com/cotes2020/jekyll-theme-chirpy/commit/0c9558de8a01e9ab795778f351a8bbf4d6b21763))

## [5.5.0](https://github.com/cotes2020/jekyll-theme-chirpy/compare/v5.4.0...v5.5.0) (2023-01-29)

### Features

* **i18n:** add Arabic translation ([#857](https://github.com/cotes2020/jekyll-theme-chirpy/issues/857)) ([765af53](https://github.com/cotes2020/jekyll-theme-chirpy/commit/765af53b77e5c63804784d5728f5970ae274c2c7))
* **i18n:** add Czech language ([#833](https://github.com/cotes2020/jekyll-theme-chirpy/issues/833)) ([98d48f5](https://github.com/cotes2020/jekyll-theme-chirpy/commit/98d48f5da412276d4a0c99cd01a87b19349bc6bc))
* **i18n:** add Finnish translations ([#843](https://github.com/cotes2020/jekyll-theme-chirpy/issues/843)) ([d6d0318](https://github.com/cotes2020/jekyll-theme-chirpy/commit/d6d03183eaf94b44e037cc48b6e1c47cee183f6e))
* **i18n:** add Italian translation ([#850](https://github.com/cotes2020/jekyll-theme-chirpy/issues/850)) ([9a011e1](https://github.com/cotes2020/jekyll-theme-chirpy/commit/9a011e14d66195d8b2fb9ec62f3e60a3e56cd032))

### Bug Fixes

*  copy command line incomplete(`.gp` part) ([41ed331](https://github.com/cotes2020/jekyll-theme-chirpy/commit/41ed33145639415148aec8e85edc7a6fd0de0ca3))
* correct encoding of spaces in share URLs ([#835](https://github.com/cotes2020/jekyll-theme-chirpy/issues/835)) ([f2d2858](https://github.com/cotes2020/jekyll-theme-chirpy/commit/f2d285844e6e2979f2b0eec1d20073d3c05b6c0c))
* post's image would cover the PWA update alert ([bd374dd](https://github.com/cotes2020/jekyll-theme-chirpy/commit/bd374dd383c50f89c8f018ecb4e25772eeb8f6d8))
* prompt with nested blockquotes renders incorrectly ([#846](https://github.com/cotes2020/jekyll-theme-chirpy/issues/846)) ([babb4a0](https://github.com/cotes2020/jekyll-theme-chirpy/commit/babb4a0c5a58ceb2e4093bc465670accdd526c18))

## [5.4.0](https://github.com/cotes2020/jekyll-theme-chirpy/compare/v5.3.2...v5.4.0) (2022-12-27)

### Features

* add `rel="me"` to Mastodon sidebar contact links for verification ([#807](https://github.com/cotes2020/jekyll-theme-chirpy/issues/807)) ([d2190c7](https://github.com/cotes2020/jekyll-theme-chirpy/commit/d2190c726f61c8c9732b88b4aecf699dc8bc7deb))
* add embed video support ([ed6dc53](https://github.com/cotes2020/jekyll-theme-chirpy/commit/ed6dc539eff7003a3765bcd8c31ae5e91a863d65))
* add shimmer background when image loads ([ab16fdc](https://github.com/cotes2020/jekyll-theme-chirpy/commit/ab16fdc7fc26811130b98a1773beb62bff6182e8))
* set preview image ratio to 1.91 : 1 ([4b6ccbc](https://github.com/cotes2020/jekyll-theme-chirpy/commit/4b6ccbcbccce27b9fcb035812efefe4eb69301cf))
* support dark and light mode images ([#481](https://github.com/cotes2020/jekyll-theme-chirpy/issues/481)) ([9306c7b](https://github.com/cotes2020/jekyll-theme-chirpy/commit/9306c7b39ecf9d9146bc1a25eebedc38eb2c3dd6))
* support LQIP for images ([bffaf63](https://github.com/cotes2020/jekyll-theme-chirpy/commit/bffaf6374f265cec96ef743d42b46fbec3b59797))

### Bug Fixes

* `hreflang` tag attribute of feed misses `site.alt_lang` ([7651d28](https://github.com/cotes2020/jekyll-theme-chirpy/commit/7651d2851b4bb7d8f0d068b62c036c89a1089bbc))
* `og:image` will be incorrect if the image uses a cross-domain URL ([8de1abd](https://github.com/cotes2020/jekyll-theme-chirpy/commit/8de1abda6be3633982392178731431b0ddb1b52b))
* refactoring error when the image URL contains parameters ([ec98f07](https://github.com/cotes2020/jekyll-theme-chirpy/commit/ec98f07aca0b80a9c07fbcdc8e0d7d66dba98ed2))
* spaces in post title are encoded when sharing ([7efd2f8](https://github.com/cotes2020/jekyll-theme-chirpy/commit/7efd2f8aa2ea1c3aeb7d740bf9a018881c26fe65))

### Improvements

* **cdn:** optimize cache policy for static assets ([7fb0ee0](https://github.com/cotes2020/jekyll-theme-chirpy/commit/7fb0ee0bedb63eee3f90a49c6d7fb8b5d78c9830))

## [5.3.2](https://github.com/cotes2020/jekyll-theme-chirpy/compare/v5.3.1...v5.3.2) (2022-11-22)

### Bug Fixes

* `mermaid` occasionally fails to initialize ([#536](https://github.com/cotes2020/jekyll-theme-chirpy/issues/536)) ([48f14e3](https://github.com/cotes2020/jekyll-theme-chirpy/commit/48f14e39ac81bbfb3b9913ea3ee789d775b2d1ae))
* **comment:** disqus doesn't follow theme mode switching ([b0d5956](https://github.com/cotes2020/jekyll-theme-chirpy/commit/b0d5956f5a0ed894984d6b1754efeba04d8bc966))
* restore full-text search ([#741](https://github.com/cotes2020/jekyll-theme-chirpy/issues/741)) ([6774e0e](https://github.com/cotes2020/jekyll-theme-chirpy/commit/6774e0e1fb37cf467b14be481347412713763f05))
* the image URL in the SEO-related tags is incomplete ([#754](https://github.com/cotes2020/jekyll-theme-chirpy/issues/754)) ([f6e9a3f](https://github.com/cotes2020/jekyll-theme-chirpy/commit/f6e9a3fccf7ab34db71f8aefaf86fdcc05861076))

## [5.3.1](https://github.com/cotes2020/jekyll-theme-chirpy/compare/v5.3.0...v5.3.1) (2022-10-25)

### Bug Fixes

* 404 page missing title in tablet/desktop view ([5511b28](https://github.com/cotes2020/jekyll-theme-chirpy/commit/5511b2883fd5a395fddfb642588d00c122f18da7))
* prompt content overflows horizontally ([#705](https://github.com/cotes2020/jekyll-theme-chirpy/issues/705)) ([fb13e32](https://github.com/cotes2020/jekyll-theme-chirpy/commit/fb13e3219b5eca0d2e4f86a1ecabfab75240369f))
* **tools:** multiple configuration files will fail the test ([80cb0b3](https://github.com/cotes2020/jekyll-theme-chirpy/commit/80cb0b371754e96772a7907877a8ce196398ba3d))

### Improvements

* **layout:** improve the min-height of main content ([#674](https://github.com/cotes2020/jekyll-theme-chirpy/issues/674)) ([49bb93c](https://github.com/cotes2020/jekyll-theme-chirpy/commit/49bb93cc0c89ad9cfaad5edcf9cb28c3d5134575))
* modify checkbox icon with `Liquid` ([1fd665b](https://github.com/cotes2020/jekyll-theme-chirpy/commit/1fd665bf4990c26ae23635c511c5abc9640184d1))
* optimize the extra padding in lists ([#703](https://github.com/cotes2020/jekyll-theme-chirpy/issues/703)) ([39da11e](https://github.com/cotes2020/jekyll-theme-chirpy/commit/39da11e3f3685f49321757576d2b87a48bf25db5)), closes [#702](https://github.com/cotes2020/jekyll-theme-chirpy/issues/702)
* **posts:** improve core block bottom padding ([d2fb98b](https://github.com/cotes2020/jekyll-theme-chirpy/commit/d2fb98b3e57f2f6c3fc3816551cd0721731adf40))
* truncate post content for search results ([647eea8](https://github.com/cotes2020/jekyll-theme-chirpy/commit/647eea8dbd716f9d3cb8330c3139fa753903f51d))
* **typography:** optimize the line height of post content ([eac3f9b](https://github.com/cotes2020/jekyll-theme-chirpy/commit/eac3f9b434ca77e3dc64eea9cedea7b93e7b306b))

### Others

* **giscus:** add `reactions-enabled` option ([#712](https://github.com/cotes2020/jekyll-theme-chirpy/issues/712)) ([70662a0](https://github.com/cotes2020/jekyll-theme-chirpy/commit/70662a0365e6b9378602dc0a57462ddad5aebcf5))
* **locale:** restore options for changing date format ([#716](https://github.com/cotes2020/jekyll-theme-chirpy/issues/716)) ([f904e8c](https://github.com/cotes2020/jekyll-theme-chirpy/commit/f904e8cd48c343cc31e25859d9d50bfe2c056f41))
* remove site config option `prefer_datetime_locale` ([6852ceb](https://github.com/cotes2020/jekyll-theme-chirpy/commit/6852ceb280927ff4e753a3e1131f2b396d9807d0))

## [5.3.0](https://github.com/cotes2020/jekyll-theme-chirpy/compare/v5.2.1...v5.3.0) (2022-09-23)

### Features

* add multiple authors to a post ([#677](https://github.com/cotes2020/jekyll-theme-chirpy/issues/677)) ([f1d9e99](https://github.com/cotes2020/jekyll-theme-chirpy/commit/f1d9e99bc02d3cd0a6b0cd1beac545f0cc7a24f8)), closes [#675](https://github.com/cotes2020/jekyll-theme-chirpy/issues/675)
* **i18n:** add Bulgarian support  ([#612](https://github.com/cotes2020/jekyll-theme-chirpy/issues/612)) ([2fed338](https://github.com/cotes2020/jekyll-theme-chirpy/commit/2fed338ce6d078bf528c9717201fbc475f88cd22))
* **i18n:** add German locale file ([#663](https://github.com/cotes2020/jekyll-theme-chirpy/issues/663)) ([940b281](https://github.com/cotes2020/jekyll-theme-chirpy/commit/940b2810e95065e30600ae8d5e4612e7183da60e))
* **i18n:** add Hungarian locale file ([#597](https://github.com/cotes2020/jekyll-theme-chirpy/issues/597), [#598](https://github.com/cotes2020/jekyll-theme-chirpy/issues/598)) ([b032977](https://github.com/cotes2020/jekyll-theme-chirpy/commit/b0329775fc24d0323e5cc04cda46ece8b4531802))
* **i18n:** add Turkish language ([#631](https://github.com/cotes2020/jekyll-theme-chirpy/issues/631)) ([ad137fa](https://github.com/cotes2020/jekyll-theme-chirpy/commit/ad137fa2945b1870b9c1dd5e9212a5f4af7c3580))

### Bug Fixes

* add missing color to linkedin icon for share list ([#683](https://github.com/cotes2020/jekyll-theme-chirpy/issues/683)) ([0dcd39d](https://github.com/cotes2020/jekyll-theme-chirpy/commit/0dcd39d491c9c49e4acf7f75f83fe6e1d1839e37))
* code contains spaces in headings ([#644](https://github.com/cotes2020/jekyll-theme-chirpy/issues/644)) ([3fa1bf3](https://github.com/cotes2020/jekyll-theme-chirpy/commit/3fa1bf305451f645a7f3aa93863b076463c8f165))
* correct spelling of `panel` ([#686](https://github.com/cotes2020/jekyll-theme-chirpy/issues/686)) ([b288587](https://github.com/cotes2020/jekyll-theme-chirpy/commit/b288587c1c3d113a1c52c2d25fb46cddda348961))
* correct the i18n for tab titles ([0c5b697](https://github.com/cotes2020/jekyll-theme-chirpy/commit/0c5b697fd3b283b6a5c926742b61ed49d8688c18))
* the `code` doesn't wrap inside the prompt ([#626](https://github.com/cotes2020/jekyll-theme-chirpy/issues/626)) ([378b65a](https://github.com/cotes2020/jekyll-theme-chirpy/commit/378b65a0617787813519dde74d6f741f255eff3d))

## [5.2.1](https://github.com/cotes2020/jekyll-theme-chirpy/compare/v5.2.0...v5.2.1) (2022-06-17)

### Bug Fixes

* exclude CHANGELOG from output ([971fe03](https://github.com/cotes2020/jekyll-theme-chirpy/commit/971fe03ec329ae49e7d60fe3af6101cfbd1acd6c))
* **PWA:** sometimes update notification is not triggered ([96af729](https://github.com/cotes2020/jekyll-theme-chirpy/commit/96af7291ea5b2c5ed6372e7b6f7725e67c69f1ba))

## [5.2.0](https://github.com/cotes2020/jekyll-theme-chirpy/compare/v5.1.0...v5.2.0) (2022-06-09)

### Features

* add es-ES support to locales ([#533](https://github.com/cotes2020/jekyll-theme-chirpy/issues/533)) ([efe75ad](https://github.com/cotes2020/jekyll-theme-chirpy/commit/efe75adf2784956afb7a0b67f6634b146d9cb03b))
* add fr-FR support to locales ([#582](https://github.com/cotes2020/jekyll-theme-chirpy/issues/582)) ([94e8144](https://github.com/cotes2020/jekyll-theme-chirpy/commit/94e81447afa457b1a6b7e8f487c47502803556d7))
* add Vietnamese locale ([#517](https://github.com/cotes2020/jekyll-theme-chirpy/issues/517)) ([171463d](https://github.com/cotes2020/jekyll-theme-chirpy/commit/171463d76da9b7bc25dd327b8f0a868ea79e388b))
* add pt-BR support to locales ([c2c503f](https://github.com/cotes2020/jekyll-theme-chirpy/commit/c2c503f63336884282b6bda4ec0703d6ae76771b))
* add option to turn off PWA ([#527](https://github.com/cotes2020/jekyll-theme-chirpy/issues/527)) ([106c981](https://github.com/cotes2020/jekyll-theme-chirpy/commit/106c981bac71e7434204a77e1f0c9c61d6eb1509))
* **PWA:** add Service Worker update notification ([d127183](https://github.com/cotes2020/jekyll-theme-chirpy/commit/d127183b9774f6321e409acdb66bf8a85d8814be))
* support showing description of preview image ([2bd6efa](https://github.com/cotes2020/jekyll-theme-chirpy/commit/2bd6efa95a174ac44e30a3af1e57e6f40d6e0e3a))

### Bug Fixes

* alt is not a valid attribute for 'a' tag ([58928db](https://github.com/cotes2020/jekyll-theme-chirpy/commit/58928dbc9068db4e4cda4371eeae1865920dce6a))
* assets URL is missing `baseurl` in self-hosted mode ([#591](https://github.com/cotes2020/jekyll-theme-chirpy/issues/591)) ([54124d5](https://github.com/cotes2020/jekyll-theme-chirpy/commit/54124d5134995fce52e4c2fc0a5d4d1743d6264d))
* correct the `twitter:creator` of Twitter summary card ([96a16c8](https://github.com/cotes2020/jekyll-theme-chirpy/commit/96a16c868ede51e7dfa412de63ffa1e5a49add7f))
* correctly URL encode share links ([4c1c8d8](https://github.com/cotes2020/jekyll-theme-chirpy/commit/4c1c8d8b0eacecbbaa2d522bbdd6430f350ff760)), closes [#496](https://github.com/cotes2020/jekyll-theme-chirpy/issues/496)
* follow paginate_path config for pagination ([6900d9f](https://github.com/cotes2020/jekyll-theme-chirpy/commit/6900d9f2bc9380cbda4babf611c6eeff345291af))
* force checkout of `gh-pages` branch ([#544](https://github.com/cotes2020/jekyll-theme-chirpy/issues/544)) ([5402523](https://github.com/cotes2020/jekyll-theme-chirpy/commit/5402523ae52a3740bcc15df0b226b2612644945d))
* horizontal scroll for long equations ([#545](https://github.com/cotes2020/jekyll-theme-chirpy/issues/545)) ([30787fc](https://github.com/cotes2020/jekyll-theme-chirpy/commit/30787fc4cf151e955bb7afc26dfd859f1a06fce6))
* p is not allowed in span ([4f590e2](https://github.com/cotes2020/jekyll-theme-chirpy/commit/4f590e2bba0639751771211bc0d357828ae70404))
* remove whitespace from avatar URL ([#537](https://github.com/cotes2020/jekyll-theme-chirpy/issues/537)) ([0542b51](https://github.com/cotes2020/jekyll-theme-chirpy/commit/0542b5149c8287dca60e37f46ee36f31b43455e4))
* resume the preview image SEO tag ([#529](https://github.com/cotes2020/jekyll-theme-chirpy/issues/529)) ([b8d1bcd](https://github.com/cotes2020/jekyll-theme-chirpy/commit/b8d1bcd3dea0abd1afef7ef154a4501fbb18938d))
* script code should be in head or body, not in between ([2103191](https://github.com/cotes2020/jekyll-theme-chirpy/commit/2103191b2faf714a8e4418c7c347a1f942b51af8))
* spurious header closing tags ([59e9557](https://github.com/cotes2020/jekyll-theme-chirpy/commit/59e955745f02f9b57c65af70b0979cd4a98bf53f))
* table bypass refactoring when it contains IAL ([#519](https://github.com/cotes2020/jekyll-theme-chirpy/issues/519)) ([5d85ccb](https://github.com/cotes2020/jekyll-theme-chirpy/commit/5d85ccb9943aac88dbbefebe1c2234cdcbae5c53))
* **theme mode:** `SCSS` syntax error ([#588](https://github.com/cotes2020/jekyll-theme-chirpy/issues/588)) ([76a1b6a](https://github.com/cotes2020/jekyll-theme-chirpy/commit/76a1b6a068c369138422dcd18ba08ec8cc3749a6))
* use `jsonify` to generate valid json ([#521](https://github.com/cotes2020/jekyll-theme-chirpy/issues/521)) ([dd9d5a7](https://github.com/cotes2020/jekyll-theme-chirpy/commit/dd9d5a7207b746342d07176d8969dc4f2c380bf2))
* when the `site.img_cdn` is set to the local path, the preview-image path loses the `baseurl` ([9cefe58](https://github.com/cotes2020/jekyll-theme-chirpy/commit/9cefe58993d9ea3a3a28424e7ffd8e0911567c5c))

### Improvements

* avoid post pageviews from shifting while loading ([135a16f](https://github.com/cotes2020/jekyll-theme-chirpy/commit/135a16f13ee783d9308669ff9a824847a73c951c))
* avoid the layout shift for post datetime ([6d35f5f](https://github.com/cotes2020/jekyll-theme-chirpy/commit/6d35f5f8da044cfad071628bb53776de03efaae4))
* **categories:** support singular and plural forms of locale ([#595](https://github.com/cotes2020/jekyll-theme-chirpy/issues/595)) ([35cadf9](https://github.com/cotes2020/jekyll-theme-chirpy/commit/35cadf969dd0161ee62503e242c545f006f7072b))
* improve the responsive design for ultrawide screens ([#540](https://github.com/cotes2020/jekyll-theme-chirpy/issues/540)) ([5d6e8c5](https://github.com/cotes2020/jekyll-theme-chirpy/commit/5d6e8c5ef6aa71b4d2600c5305f6e8ba540557f7))
