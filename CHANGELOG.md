# Changelog

All notable changes to this project will be documented in this file. See [standard-version](https://github.com/conventional-changelog/standard-version) for commit guidelines.

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
