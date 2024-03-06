---
title: CyberAttack - Social Engineering - Phishing
date: 2018-10-5 11:11:11 -0400
categories: [10CyberAttack, SocialEngineering]
tags: [SocialEngineering, Phishing]
toc: true
image:
---

> Phishing (S+ 7th ch10)

- [Phishing 网络仿冒](#phishing-网络仿冒)
  - [overview](#overview)
    - [Type of phishing](#type-of-phishing)
    - [Goals of phishing](#goals-of-phishing)
  - [New Phishing Attacks](#new-phishing-attacks)
    - [solution](#solution)
  - [Spear Phishing](#spear-phishing)
    - [solution](#solution-1)
  - [Whaling](#whaling)

---

# Phishing 网络仿冒

## overview


- sending email to users with the purpose of tricking them into revealing personal information or clicking on a link.
  - sending email tricking them into revealing personal information or clicking on a link.
  - ask for a piece of information missing, as if it is a legitimate request.
  - scam the user into surrendering private information, identity theft.
  - direct the user to website to update personal information, like password, credit card…

- Example:
  - An email like from a bank, state that there is a problem with the person’s account or access privileges.
  - One of the best counters: simply read the URL. If it is the legitimate URL.

---

### Type of phishing
- Email, SMS, telephone, etc...
- social engineering

---

### Goals of phishing

- Manipulate User

- Escalate Privileges (to more Privileges person)

- Pivot Access

- Steal Credentials
  - Credential Re-use

1. Email from Friends
   - an attacker has joined your friend’s computer to a botnet.
   - A bot herder is now using your friend’s computer to send out phishing emails.

2. Phishing to Install Malware
   - One phishing email looked like it was from a news organization with headlines of recent news events. `If the user clicked anywhere in the email`, it showed a dialog box indicating that the user’s version of Adobe Flash was too old to view the story. It then asked, “Would you like to upgrade your version of Adobe Flash?” If the user clicked Yes, it downloaded and installed malware.

3. Phishing to Validate Email Addresses
   - A simple method to validate email addresses: beacons.
   - **Beacon**:
     - a link included in the email that links to an image stored on an Internet server.
     - The link includes unique code that identifies the receiver’s email address.
   - For the email application to display the image, it must retrieve the image from the Internet server.
   - When the server hosting the image receives the request, it logs the user’s email address, indicating it’s valid.
   - This is one of the reasons that most email programs won’t display images by default.

4. Phishing to Get Money
   - This scam often requires the victim to pay a small sum of money with the promise of a large sum of money.
   - Lottery scams inform email recipients they won. Victims sometimes have to pay small fees to release the funds or provide bank information to get the money deposited.

## New Phishing Attacks
- criminals are also launching new phishing attacks.
- best way to prevent attacks: `educate people` about what the criminals are doing now.
- Example: criminals crafted a `sophisticated attack on Gmail users that fooled even tech-savvy users`.
  - captured the Gmail credentials of one user, logged on, scoured it for sent emails, attachments, and subject lines.
  - used this account to send emails to people this person previously emailed, often using similar subject lines.
  - Additionally, include a thumbnail of a document.
  - clicking the thumbnail provides a preview of the document. However, this instead opened up another tab within the browser with a URL like this: data:text/html,https://accounts.google.com/ServiceLogin?service=mail…
  - accounts.google.com, looks legitimate. Additionally, the page shows `a sign-in page that looks exactly like the Google sign-in page`.
  - Users who were tricked into “logging on” on this bogus but perfectly created web page were compromised.
  - Attackers quickly logged on to this account and started the process all over again, hoping to snare other unsuspecting users.
- In one publicized example
  - the attackers used a compromised account to resend a team practice schedule to all the members of the team.
  - It included a similar subject line and screenshot of the original attachment.
  - Some recipients who received the email clicked the thumbnail and were taken to the same URL with accounts.google.com in it.
  - `Some were tricked and entered their credentials to apparently log on to Google`.
  - Attackers quickly logged on to the newly compromised accounts and started the process again.

---

### solution

- verification/potection involve human action

- Session cookie
  - app should set session cookie


- Security token + Session cookie
  - limit:
    - meant to be temporary
    - meant to be echanged for user information only
    - one time use
    - same ip from who generate and who use it
  - enhanced Authentication flow
    - User reach app
    - app redirect user to Authentication server
    - app pass login pin to Authentication server and get returned security token
    - user login to app with security token to exchange the Session token for user info
    - app validate the cookie with Authentication server
    - app return user info to user


- Device token (tidy to HW) + Security token + Session cookie
  - limit:
    - gap:
      - create mv with Hardware ID
  - enhanced Authentication flow
    - User reach app
    - app redirect user to system chain
    - app pass login pin to system chain and get returned Device cookie
    - app ping Authentication server with Device cookie and login pin
    - Authentication server return the Session token
    - app send and store the Session token in login chain
    - app ping Authentication server with Session token
    - Authentication server return security token
    - user got security token

---


## Spear Phishing

矛 Spear Phishing
- a unique form of phishing
- **targets a specific organization**

- an e-mail spoofing fraud attempt that targets a specific organization, seeking unauthorized access to confidential data.

- the message is made like from someone you know and trust, not informal third party.
  - spear phishing messages appear to come from a trusted source. an individual within the recipient's own company, someone in a position of authority.
  - Phishing messages usually appear to come from a large and well-known company or Web site with a broad membership base, like eBay or PayPal.

- Example:
  - a message that appears to be `from your boss` telling you that there is a problem with your direct deposit account and that you need to access this HR link right now to correct it.
- works better than phishing: uses information from email databases, friends lists…

### solution
- One solution:
  - **use digital signatures**.
- The CEO and anyone else in the company can sign their emails with a digital signature.
- This provides a high level of certainty to personnel on who sent the email.


---

## Whaling

- nothing more than phishing or spear phishing, but for **big users**. target high-level executives.

- the whaler identifies one person who can gain all of the data they want, a manager or owner, and targets the phishing campaign at them.

- Example:
  - attackers singled out as many as 20,000 senior corporate executives in a fine-tuned phishing attack.
  - The emails looked like official subpoenas requiring the recipient to appear before a federal grand jury, included the executive’s full name and other details,
  - The emails also included a link for more details about the subpoena.
  - If clicked the link, it took them to a web site that indicated they needed a browser add-on to read the document.
  - If they approved this install, they actually installed a keylogger and malware.
    - Keylogger: recorded all keystrokes to a file,
    - Malware: gave the attackers remote access to the executives’ systems.


- Although not as common, some whaling attacks attempt to reach the executive via phone to get the data. However, many executives have assistants who screen calls to prevent attackers from reaching the executive via phone.






.
