# RTL Support for Jekyll Theme Chirpy

This document describes how the Right-to-Left (RTL) support works in the Chirpy theme.

## Table of Contents

- [Overview](#overview)
- [Configuration](#configuration)
- [How It Works](#how-it-works)
- [Creating RTL Content](#creating-rtl-content)
- [Language-Specific Fonts](#language-specific-fonts)
- [Customization](#customization)
- [Known Limitations](#known-limitations)

## Overview

The Chirpy theme provides support for Right-to-Left (RTL) languages such as Arabic, Hebrew, Persian, Urdu, and more. This support is implemented with minimal changes to the theme's core functionality and ensures a seamless experience for users reading content in RTL languages.

## Configuration

RTL support can be configured in your `_config.yml` file:

```yaml
# Force RTL mode for the entire website
# When set to true, all pages will be rendered right-to-left, regardless of language
# Useful if your entire site is in an RTL language
# If enabled, also set one of the RTL languages in rtl_languages list as your site.lang value
rtl_support: false

# RTL Languages array
# If you want to have your entire website in an RTL language, and you set rtl_support: true,
# then set one of these languages for lang: in the _config.yml and your entire website locale will change too.
# But if rtl_support is false and you just want specific pages in RTL with RTL locale,
# add 'lang: fa' (or another code from this list) to the front matter of those specific pages.
# The page will then be displayed in RTL mode with the appropriate font and locale
rtl_languages:
  - ar
  - fa
  - ku-IQ
  - ur-PK
  - ps-AF
  - dv-MV
```

- `rtl_support`: When set to `true`, forces RTL mode for the entire website, regardless of language. If enabled, you should also set one of the RTL languages as your main site language (e.g., `lang: fa` in `_config.yml`) to apply the appropriate locale.

- `rtl_languages`: A list of language codes that trigger RTL mode. These can be used in two ways:
  1. If `rtl_support: true`, set one of these as your site's main language in `_config.yml`
  2. If `rtl_support: false`, add `lang: fa` (or another code from this list) to the front matter of specific pages you want displayed in RTL mode

## How It Works

The RTL support works by:

1. Detecting the page language using the `lang` attribute in the front matter or the site's default language.
2. Checking if the language is in the RTL languages list (or if `rtl_support` is set to `true`).
3. Setting the `dir="rtl"` attribute on the HTML tag when appropriate.
4. Applying RTL-specific styles that override the default LTR styles.
5. Loading language-specific fonts for the detected RTL language.

## Creating RTL Content

You have two options for creating RTL content:

### 1. Individual RTL Pages

To make a specific page or post display in RTL mode:

- Add the appropriate `lang` attribute in the front matter, using one of the languages defined in `rtl_languages`:

  ```yaml
  ---
  title: عنوان المقال
  author: اسم الكاتب
  date: 2023-01-01
  lang: ar
  # other front matter...
  ---

  محتوى المقال هنا...
  ```

- The theme will automatically:
  - Display only this specific page in RTL mode
  - Use the appropriate font for the language
  - Apply the corresponding locale file if available (e.g., `_data/locales/ar.yml`)
  - The rest of your site will remain in LTR mode

This approach is ideal for multilingual sites where only some content is in RTL languages.

### 2. Entire Site in RTL

If your entire site is in an RTL language:

1. Set `rtl_support: true` in your `_config.yml`
2. All pages will be displayed in RTL mode regardless of their language setting

This approach is simpler if you don't need to mix RTL and LTR content.

## Language-Specific Fonts

The theme includes built-in support for several RTL language fonts:

- **Arabic**: Noto Sans Arabic
- **Persian (Farsi)**: Vazirmatn
- **Hebrew**: Noto Sans Hebrew
- **Urdu**: Noto Nastaliq Urdu

These fonts are automatically applied based on the page's language attribute. For example, if your page has `lang: fa`, the Vazirmatn font will be used.

## Customization

If you need to customize the RTL styles:

1. The main RTL styles are in `_sass/rtl.scss`.
2. Language-specific font definitions are in `_sass/rtl-fonts.scss`.
3. You can add additional styles to these files or create your own custom styles in your theme.

To change or add RTL fonts:

1. Modify the `rtl-fonts.scss` file to include your preferred fonts.
2. Update the font assignments for specific language codes.

## Known Limitations

- Code blocks are always displayed left-to-right (LTR) for better readability of code.
- Some third-party components or embedded content might not respect RTL layout.
- RTL mode is applied on a per-page basis based on the language attribute. If you want consistent RTL layout across your entire site, set `rtl_support: true` in your configuration.
- When switching between RTL and LTR pages, there might be a brief moment before fonts are fully loaded.
