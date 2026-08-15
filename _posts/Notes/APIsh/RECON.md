---------------

### API RECONNAISSANCE

----------------

### Postman Introduction <GeekforGeeks>

----------------

- Postman is an API(Application Programming Interface) development tool that helps to build,modify and test apis. It has the ability to make various types of HTTP requests(GET, POST, PUT, PATCH), save environments for later use, converting the API to code for various languages(like JavaScript, and Python).

-----------------

### API Development(Postman)

----------------

- Interface-:

![image](https://github.com/user-attachments/assets/6e94d1b5-5a3b-4f59-bfa3-978199f4e84f)


----------------

### Proxying Postman through Burp SUite

----------------

-  Postman's proxy setting should contain burp's host and port value-:

![image](https://github.com/user-attachments/assets/5b076a89-fdb4-4f11-a76a-ba45ed23cc0c)

- Install Burp's cert-:

![image](https://github.com/user-attachments/assets/dfbe9f00-f6f8-4932-8a4b-85006a5ff1e9)

---------------

## Creating workspaces

---------------

- Click on the highlighted part

![image](https://github.com/user-attachments/assets/0957de6c-7856-4dc6-b882-4e91e5027e34)

- Then, `create workspace`

![image](https://github.com/user-attachments/assets/520950fa-c4c5-4c65-9b05-c7fd96c22e5b)


--------------

### Creating a collection for requests

--------------

- The `add` icon-:

[image](https://github.com/user-attachments/assets/445a4cbd-3a27-48f0-9b4f-476481097d18)

- Renaming it-:

[image](https://github.com/user-attachments/assets/8bf36080-2b10-4add-ba5c-d66c1c2cb24a)


--------------

### Setting environmental variables for Postman

--------------

- Pick an environment-:

<img width="380" height="310" alt="image" src="https://github.com/user-attachments/assets/29576fe0-d353-4339-8522-dc5723f6c103" />

- Set it-:

<img width="1401" height="557" alt="image" src="https://github.com/user-attachments/assets/ba754870-0335-46b0-8faa-7051599e081e" />

- Reuse with `{{base_uri}}`-:

<img width="528" height="280" alt="image" src="https://github.com/user-attachments/assets/34adc035-8205-4dd2-ae66-005cff796f50" />


--------------

### Techniques

----------------

-  Private apis
-  Public apis
-  Partner apis

---------------

- Public apis are easily found and used by end-users.It can be entirely public and meant for authenticated users.Authentication depends on the data offered to end-users.In order to facilitate this, API providers share documentation that serves as an instruction manual for a given API. This documentation should be end-user friendly and relatively straightforward to find.
- Partner apis are created exclusively by the partner of the provider. These might be harder to find if you are not a partner. Partner APIs may be documented, but documentation is often limited to the partner.
- Private apis are limited to the organization.These APIs are often documented less than partner APIS, if at all, and if any documentation exists it is even harder to find.
- API's obvious naming schemes-:

```
/api
/v1
/v2
/v3
/rest
/swagger
/swagger.json
/docs
/doc
/graphql
/graphiql
/altair
/playground
```

- Subdomains are not left out

```
api.target-name.com
uat.target-name.com
dev.target-name.com
developer.target-name.com
test.target-name.com
```

- The use of `Content-Type` headers like `application/xml` and `application/json` can be a good indicator that you've discovered an api
- Or errors in HTTP response headers

```
{"message": "Missing Authorization token"}
```

- Information gathering via third party sources and api directories

[Github](github.com)
[Postman Explore](https://www.postman.com/explore/apis)
[APIS guru](https://apis.guru/)
[Public Apis Github Project](https://github.com/public-apis/public-apis)
[Rapid API Hub](https://rapidapi.com/search/)

----------------

### PASSIVE RECONNAISSANCE

----------------

- Google Dorking-: It is useful if you are getting too many irrelevant results on Google

Queries-:

- `inurl:"/wp-json/wp/v2/users"`<->`Finds all publicly available WordPress API user directories.`
- `intitle:"index.of" intext:"api.txt"` <-> `Finds publicly available API key files`
- `inurl:"/api/v1" intext:"index of /"` <-> `Finds potentially interesting API directories.`
- `ext:php inurl:"api.php?action="` <-> `	Finds all sites with a XenAPI SQL injection vulnerability. (This query was posted in 2016; four years later, there are currently 141,000 results.)`
- `intitle:"index of" api_key OR "api key" OR apiKey -pool` <-> `find exposed api keys`

- Git Dorking-: Regardless of whether your target performs its own development, it’s worth checking GitHub (www.github.com) for sensitive information disclosure. Developers use GitHub to collaborate on software projects. Searching GitHub for OSINT could reveal your target’s API capabilities, documentation, and secrets, such as API keys, passwords, and tokens, which could prove useful during an attack.
- Examples of GitDorking query-: `filename:swagger.json extension:.json`
- Or search for sensitive string

```
api key
apikeys
api key
authorization: Bearer
access_token
secret
token
API Key exposed
```

- Trufflehog-: TruffleHog is a great tool for automatically discovering exposed secrets. You can simply use the following Docker run to initiate a TruffleHog scan of your target's Github.

Syntax-:` trufflehog git <repo> --results=verified,unknown`

- Shodan-: Shodan is the go-to search engine for devices accessible from the internet. Shodan regularly scans the entire IPv4 address space for systems with open ports and makes their collected information public on https://shodan.io. You can use Shodan to discover external-facing APIs and get information about your target’s open ports, making it useful if you have only an IP address or organization’s name to work from. Like with Google dorks, you can search Shodan casually by entering your target’s domain name or IP addresses; alternatively, you can use search parameters like you would when writing Google queries. The following table shows some useful Shodan queries.

- Queries-:

```
hostname:sensei.com 
"content-type: application/json" <-> filter responses with json body
"content-type: application/xml"  <-> filter responses with xml body
"200 OK"  <-> successful requests
"wp-json"  <-> This will search for web applications using the WordPress API.
```

- The Wayback Machine-: It can be used to find zombie apis[active but retired endpoints that remain unknown to the devs]. Zombie APIs fall under the Improper Assets Management vulnerability on the OWASP API Security Top 10 list. Finding and comparing historical snapshots of API documentation can simplify testing for Improper Assets Management.

----------------

### Active Reconnaissance

----------------

- This form of reconnaissance involves interacting with the actual target.During this process you will be scanning systems, enumerating open ports, and finding ports that have services using HTTP. Once you have found systems hosting HTTP, you can open a web browser and investigate the web application. You could find an API being advertised to end users or you may have to dig deeper. Finally, you can scan the web app for API-related directories.

- Amass-:OWASP Amass is a command-line tool that can map a target’s external network by collecting OSINT from over 55 different sources. You can set it to perform passive or active scans. If you choose the active option, Amass will collect information directly from the target by requesting its certificate information. Otherwise, it collects data from search engines (such as Google, Bing, and HackerOne), SSL certificate sources (such as GoogleCT, Censys, and FacebookCT), search APIs (such as Shodan, AlienVault, Cloudflare, and GitHub), and the web archive Wayback.

- List data sources with `amass enum -list`

![image](https://github.com/user-attachments/assets/850ed4fe-c704-4d99-8784-525a2818faae)

- Create a `config.ini` file

```
curl https://raw.githubusercontent.com/OWASP/Amass/master/examples/config.ini >~/.config/amass/config.ini
```

- API active reconnaissance-:

`amass enum -active -d <target> |grep api`

- Directory busting with `ffuf`

------------------

### Endpoint Analysis

------------------

### Reverse engineering an api with postman and mitmweb

------------------

- Installing `mitmweb`-:

```pwsh
 pip3 install mitmproxy2swagger
 pip3 install mitmweb
```

- It uses the default burp port `8080`.To use mitmweb,close burp suite to use it.
- While connected to the proxy, run `mitm.it` to get the certificate and install it.

![image](https://github.com/user-attachments/assets/e9118525-8157-43e0-b696-a1463bade257)

- Now the requests are getting logged.

![image](https://github.com/user-attachments/assets/c3c461c3-fece-4dee-a7f0-6c07022952be)

- Installing postman-:

------------------

### Creating custom documentation with MITMweb

------------------

- Interact with the webpage feature to rack up requests
- Save it, the file name will be save as `flows`

![image](https://github.com/user-attachments/assets/99890e61-855a-474b-81d1-7eaa3e52c111)

- Sift with `mitmproxyswagger`

`sudo mitmproxy2swagger -i /Downloads/flows -o spec.yml -p http://crapi.apisec.ai -f flow`

![image](https://github.com/user-attachments/assets/30ba878d-e8b4-4728-bd2d-16da53f259cf)

- Remove the unnecessary endpoints and the ignore before it(remove with sublime text with (ctrl + shift+L) -:

![image](https://github.com/user-attachments/assets/f69e0430-852d-4ede-a413-6f78e5915cfe)

![image](https://github.com/user-attachments/assets/641e93b4-c73e-4733-a2ea-54702e29a78d)

- Run the mitm script again and add `--examples`

`sudo mitmproxy2swagger -i /Downloads/flows -o spec.yml -p http://crapi.apisec.ai -f flow --examples`

- Finally, open with `https://editor.swagger.io/` to parse it and create a user friendly documentation

![image](https://github.com/user-attachments/assets/41e45fd9-71a2-4cf0-b717-1e2a4419af65)

------------------------

### Excessive Data exposure

------------------------

- Excessive Data Exposure occurs when an API provider sends back a full data object, typically depending on the client to filter out the information that they need. From an attacker's perspective, the security issue here isn't that too much information is sent, instead, it is more about the sensitivity of the sent data. This vulnerability can be discovered as soon as you are able to start making requests. API requests of interest include user accounts, forum posts, social media posts, and information about groups (like company profiles).
- If an API provider responds with an entire data object, then the first thing that could tip you off to excessive data exposure is simply the size of the response.

![image](https://github.com/user-attachments/assets/e8547e1a-a3e9-4510-8fee-cde7a830c76f)

- This instance of Excessive Data Exposure reveals usernames, emails, IDs, and vehicle IDs all of which may prove handy in additional attacks. 


---------------

### AUTHENTICATION ATTACKS

---------------

- Classic authentication attacks are techniques that have been around such as bruteforcing and password spraying. To authenticate using this basic authentication, the consumer issues a request containing a username and password, then the provider performs a check to make sure that the combination matches records stored in a database. As we know, RESTful APIs do not maintain a state, so if the API were to leverage basic authentication across all endpoints, then a username and password would have to be issued every time.
-  The classic authentication attacks in this section include password brute-force attacks with base64 encoding, password reset brute-force, and password spraying.Bruteforce with `ffuf`-:

![image](https://github.com/user-attachments/assets/19db7576-bff3-4793-ad61-093b505e20ae)

- Password spraying with cluster bomb-:
-  A technique called password spraying can evade many of these controls by combining a long list of users with a short list of targeted passwords. Let’s say you know that an API authentication process has a lockout policy in place and will only allow 10 login attempts. You could craft a list of the nine most likely passwords (one less password than the limit) and use these to attempt to log in to many user accounts. When you’re password spraying, large and outdated wordlists like rockyou.txt won’t work. There are way too many unlikely passwords in such a file to have any success. Instead, craft a short list of likely passwords, taking into account the constraints of the API provider’s password policy, which you can discover during reconnaissance. Most password policies likely require a minimum character length, upper- and lowercase letters, and perhaps a number or special character. Use passwords that are simple enough to guess but complex enough to meet basic password requirements (generally a minimum of eight characters, a symbol, upper- and lowercase letters, and a number). The first type includes obvious passwords like QWER!@#$, Password1!, and the formula Season+Year+Symbol (such as Winter2025!, Spring2025?, Fall2025!, and Autumn2025?).
-  he real key to password spraying is to maximize your user list. The more usernames you include, the higher your odds of compromising a user account with a bad password. Build a user list during your reconnaissance efforts or by discovering excessive data exposure vulnerabilities. Let's revisit the crAPI excessive data exposure that we discovered earlier in the course. 

- Gather data from sensitive data exposure-:

 ![image](https://github.com/user-attachments/assets/cd984536-4551-43a1-a82d-ec4b95d298ea)

 - Or just use curl-:

![image](https://github.com/user-attachments/assets/51632378-bb22-47ca-9d49-b0c6a5b4f520)

- Sort it out with `uniq` and `sort -u`

 ![image](https://github.com/user-attachments/assets/b705fb36-63dd-4bdd-a52d-4d2bde93608a)

- Bruteforce with Cluster Bomb-:

![image](https://github.com/user-attachments/assets/5cf763ed-b49f-4d6e-9317-eda85b5e5531)

--------------------

### Passwords in requests can also be base64 encoded

--------------------

- It can can achieved with `Payload processing`-`Encode`-`Base64-encode`

![image](https://github.com/user-attachments/assets/51c5aa7a-b6a6-445f-b88a-3958383e6bb1)

![image](https://github.com/user-attachments/assets/12f6f8dd-a99f-4aee-80d1-f46baf482717)

- Proof-:

![image](https://github.com/user-attachments/assets/232f403a-c934-4ffd-8536-3e9b69a0cca1)

------------------------

### Exploiting token flaws with `jwt_tool.py`

------------------------

- In order to study jwt token with sequencer,navigate to the Sequencer tab and select the request that you forwarded. Here we can use the Live Capture to interact with the target and get live tokens back in a response to be analyzed. To make this process work, you will need to define the custom location of the token within the response. Select the Configure button to the right of Custom Location. Highlight the token found within quotations and click OK.

![image](https://github.com/user-attachments/assets/6f548e73-3b5c-401a-9ede-23e11a53d023)

- Once it has been defined, you should press the `start live capture`.Using Sequencer against crAPI shows that the tokens generated seem to have enough randomness and complexity to not be predictable. Just because your target sends you a seemingly complex token, does not mean that it is safe from token forgery. Sequencer is great at showing that some complex tokens are actually very predictable. If an API provider is generating tokens sequentially then even if the token were 20 plus characters long, it could be the case that many of the characters in the token do not actually change. Making it easy to predict and create our own valid tokens.

![image](https://github.com/user-attachments/assets/e6588d27-4e36-49b4-b722-854955122d9f)

------------------

### JWT ATTACKS

-------------------

- "none" attack
- Algorithm confusion

-------------------

### Automating jwt attacks with `JWT_TOOL`

-------------------

- Use the playbook test to interact with the site.
- Match and replace in  burp suite for a payload

![image](https://github.com/user-attachments/assets/7854ff0c-64b7-4186-bd25-bd6a4881e0e0)

- For the username, use the regex `,.*`

  ![image](https://github.com/user-attachments/assets/52036b02-e159-44c5-ab53-67fedac91a66)

- For the password, use the regex `.*,`

![image](https://github.com/user-attachments/assets/01bfc19a-2832-45d2-9284-05c3e6da66ec)

---------------

### JWT TOOL

---------------
- Error `no proxy` fix-: Use `-np` for no proxy

 ![image](https://github.com/user-attachments/assets/3e117e95-317a-47d6-a92b-0ef31ba6756f)

- Use the playbook test to scan for common jwt vulnerabilities,syntax-:

`python3 jwt_tool.py -t http://crapi.apisec.ai/identity/api/v2/user/dashboard -rh "Authorization: Bearer eyJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJiaW1iaW16QGdtYWlsLmNvbSIsImlhdCI6MTc0MzUxMDEwNSwiZXhwIjoxNzQzNTk2NTA1fQ.OufYUVmq7Ys8Dn8FSTFNQLMDy3ZKRA3YAzuLzb0J4na0PDea4afHLeQhSbuXOUpethOxue5Z4GqtyDIaQPNTSA" -M pb -np`

------------------- 

### Broken Authorization

--------------------

- It focuses on two categories `Broken Object Level Authorization` and `Broken Function Level Authorization`

- An API’s authentication process is meant to validate that users are who they claim to be. An API's authorization is meant to allow users to access the data they are permitted to access. In other words, UserA should only be able to access UserA's resources and UserA should not be able to access UserB's resources. API providers have been pretty good about requiring authentication when necessary, but there has been a tendency to overlook controls beyond the hurdle of authentication. Authorization vulnerabilities are so common for APIs that the OWASP security project included two authorization vulnerabilities on its top ten list, Broken Object Level Authorization (BOLA) and Broken Function Level Authorization (BFLA).
-  BOLA vulnerabilities occur when an API provider does not restrict access to access to resources.
-  BFLA vulnerabilities are present when an API provider does not restrict the actions that can be used to manipulate the resources of other users. BOLA is the ability for UserA to see UserB's bank account balance and BFLA is the ability to for UserA to transfer funds from UserB's account back to UserA.
-  Ingredients for BOLA-:
   - Resource ID-: a resource identifier will be the value used to specify a unique resource. This could be as simple as a number, but will often be more complicated.
   - Requests that access resources. In order to test if you can access another user's resource, you will need to know the requests that are necessary to obtain resources that your account should not be authorized to access.
   - Missing or flawed access controls. In order to exploit this weakness, the API provider must not have access controls in place. This may seem obvious, but just because resource IDs are predictable, does not mean there is an authorization vulnerability present.

- Finding resource id-: You can test for authorization weaknesses by understanding how an API’s resources are structured and then attempting to access resources you shouldn’t be able to access. By detecting patterns within API paths and parameters, you might be able to predict other potential resources. The bold resource IDs in the following API requests should catch your attention

- Examples-:You can proceed to tweak the bold ids

```
GET /api/resource/1
GET /user/account/find?user_id=15
POST /company/account/Apple/balance
POST /admin/pwreset/account/90
```

- In these simple examples, you’ve performed an attack by merely replacing the bold items with other numbers or words. If you can successfully access the information you shouldn’t be authorized to access, you have discovered an authorization vulnerability. Here are a few ideas for  requests that could be good targets for an authorization test.Most times you can leverage on a excessive data exposure to read data.

![image](https://github.com/user-attachments/assets/a989b63f-311e-4bca-b365-e46f90f553a5)

-----------------

### Broken Function Level Authorization

------------------

- Where BOLA is about acessing resources that is not yours, BFLA is about performing unauthorized actions on resources that are not yours.These requests could be lateral actions or escalated actions. Lateral actions are requests that perform actions of users that are the same role or privilege level. Escalated actions are requests that perform actions that are of an escalated role like an administrator. The main difference between hunting for BFLA is that you are looking for functional requests. This means that you will be testing for various HTTP methods, seeking out actions of other users that you should not be able to perform.If you think of this in terms of a social media platform, an API consumer should be able to delete their own profile picture, but they should not be able to delete other users' profile pictures. The average user should be able to create or delete their own account, but they likely shouldn't be able to perform administrative actions for other user accounts.
- The main difference between BOLA and BFLA is that we are looking for functional requests(CRUD)- Create,Read,Update and Delete. BFLA will mainly concern requests that are used to update, delete, and create resources that we should not be authorized to. For APIs that means that we should scrutinize requests that utilize POST, PUT, DELETE, and potentially GET with parameters.


------------------

### Improper Assets Management

-----------------

- Testing for this involves checking for outdated and unsupported versions of an api.Often times an API provider will update services and the newer version of the API will be available over a new path like the following .

```
api.target.com/v3
/api/v2/accounts
/api/v3/accounts
/v2/accounts
```
- Api versioning can also be made with an header-:

```
Accept: version=2.0
Accept api-version=3
```

- Also in `GET` and `POST` data

```
/api/accounts?ver=2
POST /api/accounts

{
"ver":1.0,
"user":"hapihacker"
}
```

- In these instances, earlier versions of the API may no longer be patched or updated. Since the older versions lack this support, they may expose the API to additional vulnerabilities. For example, if v3 of an API was updated to fix a vulnerability to injection attacks, then there are good odds that requests that involve v1 and v2 may still be vulnerable.
- Non-production versions of an API include any version of the API that was not meant for end-user consumption. Non-production versions could include:
```
api.test.target.com
api.uat.target.com
beta.api.com
/api/private
/api/partner
/api/test
```

---------- 

### Mass Assignment

-------------

- Mass Assignment vulnerabilities are present when an attacker is able to overwrite object properties that they should not be able to. A few things need to be in play for this to happen. An API must have requests that accept user input, these requests must be able to alter values not available to the user, and the API must be missing security controls that would otherwise prevent the user input from altering data objects. The classic example of a mass assignment is when an attacker is able to add parameters to the user registration process that escalate their account from a basic user to an administrator. The user registration request may contain key-values for username, email address, and password. An attacker could intercept this request and add parameters like "isadmin": "true". If the data object has a corresponding value and the API provider does not sanitize the attacker's input then there is a chance that the attacker could register their own admin account.
- Queries and values to test-:

```
"isadmin": true,
"is_admin":"true",
"admin": 1,
"admin":true
```

-----------------

### API PATCH Method

-----------------

- `PATCH` standard [format](https://medium.com/@isuru89/a-better-way-to-implement-http-patch-operation-in-rest-apis-721396ac82bf)

```query
[
   { "op": "test", "path": "/a/b/c", "value": "foo" },
   { "op": "remove", "path": "/a/b/c" },
   { "op": "add", "path": "/a/b/c", "value": [ "foo", "bar" ] },
   { "op": "replace", "path": "/a/b/c", "value": 42 },
   { "op": "move", "from": "/a/b/c", "path": "/a/b/d" },
   { "op": "copy", "from": "/a/b/d", "path": "/a/b/e" }
]
```
------------------















 




