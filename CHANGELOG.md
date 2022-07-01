# Changelog

All notable changes to this project will be documented in this file. See [standard-version](https://github.com/conventional-changelog/standard-version) for commit guidelines.

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
