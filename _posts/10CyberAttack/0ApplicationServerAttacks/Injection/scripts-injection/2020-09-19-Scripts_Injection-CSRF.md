---
title: Meow's CyberAttack - Application/Server Attacks - Scripts Injection - CSRF
# author: Grace JyL
date: 2020-09-19 11:11:11 -0400
description:
excerpt_separator:
categories: [10CyberAttack]
tags: [CyberAttack]
math: true
# pin: true
toc: true
# image: /assets/img/sample/devices-mockup.png
---

- [CSRF - Cross-Site Request Forgery](#csrf---cross-site-request-forgery)
	- [CSRF](#csrf)
	- [CSRF In Action](#csrf-in-action)
	- [Vectors for CSRF Attack](#vectors-for-csrf-attack)
	- [CSRF Protection Mechanisms](#csrf-protection-mechanisms)

Book: S+ 7th ch9

---

# CSRF - Cross-Site Request Forgery

---

## CSRF

An action can consist of purchasing items, transferring monies, administering users, and managing records.

- For each action there is a corresponding `GET` or `POST` request that communicates this action from the client browser to the server.

- As many of these actions are sensitive in nature, most web applications require that the user is authenticated and that the communication channel is encrypted, i.e. HTTPS.

CSRF
- Cross-Site Request Forgery 伪造 (XSRF) CSRF 跨站请求伪造

- Known as session riding, one-click attack

- CSRF is an attack that requires two elements:
  1) a web application that performs actions
  2) an authenticated user.

- <font color=OrangeRed> involves unauthorized commands </font> coming from a trusted user of website
- 挟制用户在当前已登录的Web应用程序上执行非本意的操作的攻击方法
- forces an end user to execute unwanted actions on a web application in which they’re currently authenticated.

  - For most sites, browser requests automatically include any credentials associated with the site, such as the user’s session cookie, IP address, Windows domain credentials, and so forth.
  - Therefore, if the user is currently authenticated to the site, the site will have no way to distinguish between the forged request sent by the victim and a legitimate request sent by the victim.


- tricks the victim into submitting a malicious request, to perform an undesired function

  - 完成一些违背用户意愿的请求（如恶意发帖，删帖，改密码，发邮件等）。
  - 冒充用户发起请求（不知情 ）<font color=OrangeRed> tricks a user into performing an action on a web site </font>.

  - If the victim is a normal user, a successful CSRF attack can force the user to perform state changing requests like transferring funds, changing their email address, and so forth.
  - If the victim is an administrative account, CSRF can compromise the entire web application.


- The attacker creates a specially crafted HTML link and the user performs the action without realizing it.

- <font color=LightSlateBlue> If the web site support any action via an HTML link </font>, and attack is possible.

- Web sites typically won’t allow these actions without users first logging on.
  - But, if users have logged on before
  - Authentication information is stored on their system either in a cookie or in the web browser’s cache.
  - Some web sites automatically use third info to log users on.

- In some case, the XSRF attack allows the attacker to access the users password.

A session fixation attack is somewhat similar to CSRF.
- The attacker logs in to a legitimate site and pulls a session ID, then sends an e-mail with a link containing the fix session ID.
- When the user clicks it and logs into the same legitimate site, the hacker can now log in and run with the user’s credentials.

![page181image36093584](/assets/img/page181image36093584.jpg)


summary of a CSRF attack.

Method Type | Attack Details
---|---
Spoofing| The attacker needs to figure out the exact invocation of the targeted malicious action and then craft a link that performs the said action. Having the user click on such a link is often accomplished by sending an email or posting such a link to a bulletin board or similar message system.

So how does this attack work?
- example:
- logged into your banking website, called `ABCBank.com`.
- The bank adheres to the principle of two factor authentication (`username` and `password` and a subsequent `PIN`) and the communication between you and the bank is encrypted (via HTTPS).

- After the browser recognizes and validates the certificate issued by the bank you are logged in and viewing your information in a secure session.

  - a. Banking website requesting credentials (1st factor of authentication);

  - b. Banking website asking for personal PIN (2ndfactor of authentication);

  - c. Communication is in a secure session (https);

  - d. The Lock symbol indicates the certificate information from the banking website is valid and authenticate.

- Now that you are authenticated to the banking website and authorized to access your account, <font color=LightSlateBlue> the credential information (generally represented by a Session Identifier) is cached on the local machine </font>, usually in the <font color=OrangeRed> form of an encrypted cookie </font>.

  - The cookie will act on your behalf when credential information is repeatedly requested as you move through the website, thereby not requiring you to type your credential information repeatedly for each page you visit.

  - While this is a convenience to you, <font color=OrangeRed> this is where the CSRF attack takes advantage of this convenience </font>, combined with the trusted nature the application gives to the process: in other words, the application fails at the cliché “trust but verify.”


---

## CSRF In Action

Example:
- chatting through Facebook.
  - sends a link of funny video.
  - clicks the link, but it actually brings up Evan’s bank account information in another browser tab, takes a screenshot of it, closes the tab, and sends the information to Spencer.
  - The reason the attack is possible is because Evan is a trusted user with his own bank.
  - In order for it to work, Evan would need to have recently accessed that website and have a cookie not yet expire.

Example:
- how HTML links create action
  - consider this HTML link: `http://www.google.com/search?q=Success`. If users click this link, it works just as if the user browsed to Google and entered Success as a search term. The `?q=Success` part of the query causes the action.
  - Many web sites use the same type of HTML queries to perform actions.

Example:
  - a web site that supports user profiles.
  - If users change profile information, they log on, make the change, and click a button.
  - The web site may use a link like this to perform the action:
`http://getcertifiedgetahead.com/edit?action=set&key=email&value=you@home.com`.
  - Attackers use this knowledge to create a malicious link.
  - example, the following link could change the email address in the user profile, redirecting the user’s email to the attacker: `http://getcertifiedgetahead.com/edit?action=set&key=email&value=hacker@hackersrs.com`.


Example:

1. A user logs into his/her bank account;

1. the bank account `uses URL parameters to pass a unique identifier` for the bank account number and the type of view;

1. the user clicks a link in an email message that he believes is from his friend;

1. the link takes him to a malicious site that `exposes the URL parameters of the banking website to perform actions on behalf of the user without the user’s knowledge`.



- The following URL is used by the banking site to determine navigation and action:
`https://www.somebank.com/inet/sb_bank/BkAccounts?target=AccountSummary&currentaccountkey=encryptedec117d8fd0eb30ab690c051f73f4be34&TransferView=TRUE`

- The Request information for the above URL is as follows (the real values are removed or replaced with fake values where applicable):

	```
	Accept: text/html, application/xhtml+xml, */*

	Referer: https://www.somebank.com/inet/sb_bank/BkAccounts?target=AccountSummary& currentaccountkey=encryptedec117d8fd0eb30ab690c051f73f4be34&
	TransferView=TRUE

	Accept-Language: en-US

	User-Agent: Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0)

	Accept-Encoding: gzip, deflate

	Host: www.somebank.com

	Connection: Keep-Alive

	Cookie: JSESSIONID={value}; BrowserNavData=true|-1; somebank.com.uniqueId={value}; somebank.com.machine.session={value}; SSID={value}; SSRT={value}; SSPV={value}; UASK=39bwcDrir8moz_f8p6JftTH9hWt6EEhWpqSct35zzsfv86wySvpnVPA; somebank.com.machine.ident={value}; VisitorId=AIBJLR221KWGQYKERWP5C20120205; grpId=7; MemberGlobalSession=2:1000:5ZJBAM5213M3C515PLAR; TDO_RANDOM_COOKIE=97890180120120205153123; dcenv=1; LtpaToken2={value}=; LtpaToken={value}
	```

- The user then receives an email asking him to check out his items on an auction site at the following URL: `Http://www.somecoolacutionsite.com/sampleauction.html`

- Unknown to the user, the email was not from his friend, and when he clicks on the URL, the auction site does not contain any auctions. However, what <font color=OrangeRed> the “auction” site did was use CSRF to perform an action on behalf of the user to the banking site the user is still logged into </font>.

- Here is the HTML code from the “auction” site:

	```html
	<html>
	<head></head>
	<body>

	Welcome to the “auction” portal. Buyer beware!

	<Iframe src=”https://www.somebank.com/inet/sb_bank/BkAccounts ?target=AccountSummary&currentaccountkey=encryptedec117d8fd0eb30ab690c051f73f4be34&TransferView=TRUE” id=”xsrf” style=”width:0px; height:0px; border:0px”></iframe>

	</body>
	</html>
	```

- And the malicious Request information is as follows:

	```
	HTTP/1.0
	Accept: text/html, application/xhtml+xml, */*
	Referer: http://www.malicioussite.com/sampleauction.html
	Accept-Language: en-US
	User-Agent: Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0)
	Accept-Encoding: gzip, deflate
	Host: www.somebank.com
	Connection: Keep-Alive

	Cookie: JSESSIONID={value}; SSLB=1; SSSC=1.G5704896267906605088.7|10.607; BrowserNavData=true|-1; somebank.com.uniqueId=MTIgISEgITQwJjM2MDM3OTk0; somebank.com.machine.session=9DUvMKuboaOuRCYdLlct6Nm; UASK=39bwcDrir8moz_f8p6JftTH9hWt6EEhWpqSct35zzsfv86wySvpnVPA; MemberGlobalSession={value}; TDO_RANDOM_COOKIE={value}; dcenv=1; LtpaToken2={value}=; LtpaToken={value}
	```

- Notice something different between the two?
  - The <font color=OrangeRed> Referrer </font> between the two Requests is different
  - also, <font color=OrangeRed> all of the session and unique ID information were the same </font>.

---

## Vectors for CSRF Attack

Cross-Domain Request Type:

`<Iframe>`
- allows for the cross-domain generation of content within the current domain’s session.

`<img src=…>`
- Image tags allow pulling the src image from other domains.
- Since the browser does not validate the request is an actual image, any valid URL pointing to any location and resource can be placed in the src attribute.

`<script src=…>`
- Script tags allow pulling the src script file from other domains. Since the browser does not validate the request is an actual script, any valid URL pointing to any location and resource can be placed in the src attribute.

`<FORM ….>`
- `<iframe>, <img>, and <script>` are considered GET request methods.
- POST methods submit a `<FORM>` with input variables with a name and value attribute, to a URL specified in the action attribute of `<FORM>`. The `<FORM>` is then submitted when the user lands on the malicious page.

`Ajax`
- Taking advantage of the `<FORM>` POST method, an Ajax-based site can be sent information in an XML stream, as an example.*

`JSON`
- Many applications are developed using a JSON stream to exchange information between clients and servers. Because of the way a JSON string is formatted, it is possible to once again use the `<FORM>` request type to carefully craft a JSON type string and send the action URL. JSON strings take the form of {“field”: “value”, …}. An input field in a `<FORM>` can be used to by placing the {“field”: “value”,…} pair as the name attribute value and setting value=’no’.*

> `<FORM>` is not the only method to use. It is just easier to explain the attack using `<FORM>`.
> Also, using `<FORM>, <img>, and <script>` will allow a read-only access to the data. Information access is possible using a combination of `<script> and Ajax`. However, this exercise is left to the reader.

---

## CSRF Protection Mechanisms

- CSRF is difficult to detect with static analysis products, and only a handful of dynamic scanners can detect the possibility of a CSRF lurking within.

- The most effective strategy for detecting CSRF is to manually test the application by creating a page with one of the Cross-Domain Request Types and point the src of one of those types to your site.

- For cross-site scripting, <font color=OrangeRed> primary burden of protection: web site developers </font>.

  - Developers need to be aware of XSRF attacks and the different methods used to protect against them.
  - use dual authentication and force the user to manually enter credentials prior to performing actions.
  - expire the cookie after a short period, such as after 10 minutes, preventing automatic logon for the user.

- Implementation

  - The user must be prompted to confirm an action every time for actions concerning potentially sensitive data. The confirmation along with the design approach of uniquely identifying requests and actions will thwart phishing and related attacks.

  - All requests must be checked for the appropriate authentication token as well as authorization in the current session context.

  - Check the Referral and make sure it is generated from the target page residing in the same domain.

  - For XML and JSON verify and validate the Content Type.

- Design

  - <font color=OrangeRed> checking the HTTP Referer header </font> can also be used to validate an incoming request was actually one <font color=LightSlateBlue> from an allowed or authorized domain </font>.

  - <font color=OrangeRed> Use a unique identifier </font> to associate a user request with a specific action. The identifier should be recreated for every request and action. The identifier is considered invalid if it arrived with a request without the associated action. An example is the <font color=LightSlateBlue> use of a token that is attached to each request/action </font>.

  - CSRF attacks can be mitigated by configuring a web server to <font color=OrangeRed> send random challenge tokens </font>.

    - If every user request includes the challenge token, it becomes easy to spot illegitimate requests not initiated by the user.

  - Many programming languages support <font color=OrangeRed> XSRF tokens </font>.
	- Python and Django, two popular web development languages, require the use of an `XSRF token` in any page that includes a form, these languages call them `CSRF tokens`.
	- This token: a large random number generated each time the form is displayed.
	- When a user submits the form, the web page includes the token along with other form data.
	- web application then verifies that the token in the HTML request is the same as the token included in the web form.
	- The HTML request might look something like this: `getcertifiedgetahead.com/edit?action=set&key=email&value=you@home.com&token=1357924`
	- The token is typically much longer.
	- If the website receives a query with an incorrect error, it typically raises a 403 Forbidden error.
	- Attackers can’t guess the token, so they can’t craft malicious links that will work against the site.

  - <font color=OrangeRed> disable the running of scripts (and browser profiles) </font>.





---
