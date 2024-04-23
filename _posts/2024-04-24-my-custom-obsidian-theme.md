---
published: true
date: 2024-04-24
title: My custom Obsidian theme
---
Font: [JetBrains Mono](https://www.jetbrains.com/lp/mono)

![](/media/2024-04-24%2002.07.00.jpg)

```css
@font-face {
  font-family: "JetBrains Mono";
  src: url("webfonts/JetBrainsMono-Regular.woff2") format("woff2");
  font-weight: normal;
  font-style: normal;
}

@font-face {
  font-family: "JetBrains Mono";
  src: url("webfonts/JetBrainsMono-Italic.woff2") format("woff2");
  font-weight: normal;
  font-style: italic;
}

@font-face {
  font-family: "JetBrains Mono";
  src: url("webfonts/JetBrainsMono-Bold.woff2") format("woff2");
  font-weight: bold;
  font-style: normal;
}

@font-face {
  font-family: "JetBrains Mono";
  src: url("webfonts/JetBrainsMono-BoldItalic.woff2") format("woff2");
  font-weight: bold;
  font-style: italic;
}

:root {
  --header-color-1: red;
  --header-color-2: orange;
  --header-color-3: green;
  --header-color-4: blue;
  --header-color-5: purple;
  --header-color-6: #ff7fff;
  --background-primary: #fff;
  --link-color: blue;
}

@media (prefers-color-scheme: dark) {
  :root {
    --header-color-1: #ff7f7f;
    --header-color-2: #ffbf7f;
    --header-color-3: #7fff7f;
    --header-color-4: #7fbfff;
    --header-color-5: #bf7fff;
    --header-color-6: pink;
    --background-primary: #333;
  }
}

/* Text */

* {
  font-family: "JetBrains Mono", "Courier New", monospace;
}

a,
.cm-url {
  text-decoration: none !important;
}

.cm-strong {
  color: var(--header-color-1);
}

a:hover,
.cm-url:hover {
  text-decoration: underline !important;
}

/* End text */

/* Heading */

.cm-formatting-header + .cm-header {
  box-shadow: none;
  padding-left: 0;
}

.cm-widgetBuffer + .cm-header-1::before,
.cm-formatting-header-1::before,
h1::before {
  content: "";
  display: inline-block;
  box-shadow: inset var(--header-color-1) 3px 0;
  width: 15px;
  height: 0.75em;
}
.cm-header-1,
h1 {
  color: var(--header-color-1) !important;
}

.cm-widgetBuffer + .cm-header-2::before,
.cm-formatting-header-2::before,
h2::before {
  content: "";
  display: inline-block;
  box-shadow: inset var(--header-color-2) 3px 0,
    inset var(--background-primary) 6px 0, inset var(--header-color-2) 9px 0;
  width: 21px;
  height: 0.75em;
}
.cm-header-2,
h2 {
  color: var(--header-color-2) !important;
}

.cm-widgetBuffer + .cm-header-3::before,
.cm-formatting-header-3::before,
h3::before {
  content: "";
  display: inline-block;
  box-shadow: inset var(--header-color-3) 3px 0,
    inset var(--background-primary) 6px 0, inset var(--header-color-3) 9px 0,
    inset var(--background-primary) 12px 0, inset var(--header-color-3) 15px 0;
  width: 27px;
  height: 0.75em;
}
.cm-header-3,
h3 {
  color: var(--header-color-3) !important;
}

.cm-widgetBuffer + .cm-header-4::before,
.cm-formatting-header-4::before,
h4::before {
  content: "";
  display: inline-block;
  box-shadow: inset var(--header-color-4) 3px 0,
    inset var(--background-primary) 6px 0, inset var(--header-color-4) 9px 0,
    inset var(--background-primary) 12px 0, inset var(--header-color-4) 15px 0,
    inset var(--background-primary) 18px 0, inset var(--header-color-4) 21px 0;
  width: 33px;
  height: 0.75em;
}
.cm-header-4,
h4 {
  color: var(--header-color-4) !important;
}

.cm-widgetBuffer + .cm-header-5::before,
.cm-formatting-header-5::before,
h5::before {
  content: "";
  display: inline-block;
  box-shadow: inset var(--header-color-5) 3px 0,
    inset var(--background-primary) 6px 0, inset var(--header-color-5) 9px 0,
    inset var(--background-primary) 12px 0, inset var(--header-color-5) 15px 0,
    inset var(--background-primary) 18px 0, inset var(--header-color-5) 21px 0,
    inset var(--background-primary) 24px 0, inset var(--header-color-5) 27px 0;
  width: 39px;
  height: 0.75em;
}
.cm-header-5,
h5 {
  color: var(--header-color-5) !important;
}

.cm-widgetBuffer + .cm-header-6::before,
.cm-formatting-header-6::before,
h6::before {
  content: "";
  display: inline-block;
  box-shadow: inset var(--header-color-6) 3px 0,
    inset var(--background-primary) 6px 0, inset var(--header-color-6) 9px 0,
    inset var(--background-primary) 12px 0, inset var(--header-color-6) 15px 0,
    inset var(--background-primary) 18px 0, inset var(--header-color-6) 21px 0,
    inset var(--background-primary) 24px 0, inset var(--header-color-6) 27px 0,
    inset var(--background-primary) 30px 0, inset var(--header-color-6) 33px 0;
  width: 45px;
  height: 0.75em;
}
.cm-header-6,
h6 {
  color: var(--header-color-6) !important;
}

/* End heading */

.pdf-embed .pdf-viewer-container {
  overflow: hidden;
}

.media-embed {
  display: flex !important;
  justify-content: center;
}
```