// ==UserScript==
// @name         AWS SSO Login Request Authorizer
// @namespace    http://tampermonkey.net/
// @version      1.0
// @description  Clicks the "Allow" button automatically when running `aws sso login`
// @author       Marc PEREZ
// @match        https://*.awsapps.com/start/user-consent/authorize.html?*
// @match        https://*.awsapps.com/start/user-consent/login-success.html
// @icon         https://d2q66yyjeovezo.cloudfront.net/icon/b5164fbdf0a4526876438e688f5e4130-8f4c3d179652d29309b38012bd392a52.svg
// @grant        window.close
// ==/UserScript==
// Change this to `false` if you don't want to close the tab automatically
const CLOSE_AFTER_ALLOW = true;
(function() {
'use strict';
// Check if we successfully authorized to close the tab
if (CLOSE_AFTER_ALLOW && window.location.pathname == '/start/user-consent/login-success.html') {
window.close();
}
// Find the "Allow" button and click it
const allow_button = document.getElementById('cli_login_button');
if (allow_button) {
allow_button.click();
} else {
console.error(`${GM_info.script.name}: Allow button not found.`);
}
})();
