// ==UserScript==
// @name         Quip Style Change
// @namespace    http://tampermonkey.net/
// @version      0.1
// @description  Reduce bottom margin on section elements to make grouping of content more implicit.
// @author       Andy Richardson
// @match        https://your-quip-domain.com/* <- SWAP THIS
// @icon         https://www.google.com/s2/favicons?sz=64&domain=quip-apple.com
// @grant        none
// ==/UserScript==

(function() {
    'use strict';
    const style = document.createElement("style");
    style.innerHTML = `
      .section[aria-level] {
        margin-bottom: 5px !important;
      }

      /* Typography - modular scale */
      .section[aria-level="4"] > * {
        font-size: 14px !important;
      }
      .section[aria-level="3"] > * {
        font-size: 18.2px !important;
      }
      .section[aria-level="2"] > * {
        font-size: 23.66px !important;
      }
      .section[aria-level="1"] > * {
        font-size: 30.758px !important;
      }
      /* Heading spacing - modular scale */
      .section[aria-level="3"] {
        margin-top: 26px !important;
      }
      .section[aria-level="2"] {
        margin-top: 33.8px !important;
      }
      .section[aria-level="1"] {
        margin-top: 43.94px !important;
      }

      /* Blockquote spacing */
      .section[aria-roledescription="Blockquote"] {
        margin-top: 10px !important;
      }
    `;
    document.body.appendChild(style);
})();
