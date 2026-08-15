---------------

### CHRONOS-CTF

---------------

<img width="928" height="379" alt="image" src="https://github.com/user-attachments/assets/2ff9818a-97d4-4a1d-a0f5-0f5d4aaf5dd0" />


---------------

### Challenges

-----------------

- Web-:
 - Requests
 - ChatBot-v1
 - ChatBot-V2
 - Chatbot-v3
 - Extension

-----------------

### ChatBot-V1

-----------------

<img width="490" height="782" alt="image" src="https://github.com/user-attachments/assets/d186eab6-43b9-4089-a07c-9b0b616937c6" />

-----------------

- I checked the source and noticed a minified js file.

<img width="1548" height="662" alt="image" src="https://github.com/user-attachments/assets/7e482665-c821-4ef0-9f9d-d27acf2d48bb" />

- I beautified the source code and read through it.The variable highlighted points to an hidden html file.

<img width="1135" height="152" alt="image" src="https://github.com/user-attachments/assets/f24d5b24-6d5d-4a9f-9689-0c9fc4b1b23e" />

- Devtools-:

<img width="438" height="84" alt="image" src="https://github.com/user-attachments/assets/8a392001-0c49-4a1c-9fbc-d0a3512bf217" />

- Found the flag in the page-:`CSCTF{r0le_manag3d_vi4_localStorage_1s_b4d}`

<img width="685" height="206" alt="image" src="https://github.com/user-attachments/assets/f9513899-f532-468c-b293-57aa741baa04" />

--------------

### ChatBot-v2

--------------

<img width="508" height="876" alt="image" src="https://github.com/user-attachments/assets/357318db-19a6-43dd-8cb9-2e7502c5ad3b" />

--------------

- I checked the javascript and noticed the javascript code that the app uses to load conversations.The code uses hash type `md5` to hash an integer and passes it to endpoint `/load_conversation` to load conversations.The code snippet is vulnerable to IDOR.The mode of creating IDs is weak and can easily be replicated leading to access to other users' conversations.

```js
unction loadConversation(conversationId) {
            currentConversationId = md5sum(conversationId.toString()); 
            fetch(`/load_conversation`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ conversation_id: currentConversationId })
            })
            .then(response => response.json())
            .then(data => {
                const chatBox = document.getElementById('chat-box');
                chatBox.innerHTML = ''; 
```
- I generated md5 ids with python3.

```python3
>>> import hashlib
>>> for i in range(100): print(hashlib.md5(str(i).encode()).hexdigest())
```

- Fuzzed with burpsuite and got the flag-:`CSCTF{y0u_c4n7_h1d3_fr0m_1D0R}`

<img width="1492" height="695" alt="image" src="https://github.com/user-attachments/assets/baeb9caf-5733-4b5e-8cea-d88846f56e39" />

--------------

### Chatbot-v3

--------------

<img width="482" height="702" alt="image" src="https://github.com/user-attachments/assets/d45f0fd4-c9ad-4294-a650-41cdd22ab82f" />

--------------

- In this version, the dev has patched up the idor issue by generating guids/uuids.

<img width="1509" height="818" alt="image" src="https://github.com/user-attachments/assets/b10c47b1-154c-4929-8ff0-9970b3498a91" />

- But, it is vulnerable to sql injection.I tested based on True and False Statements to finalize that it is vulnerable to blind sqli injection.True statement reaction reveals the values of  the identifier-:

<img width="1559" height="528" alt="image" src="https://github.com/user-attachments/assets/fae4f0bf-cdfb-4669-bfb6-b1af38387fbf" />

- It appears otherwise in the false statement-:

<img width="1527" height="469" alt="image" src="https://github.com/user-attachments/assets/c15a71d1-193b-41bd-a1ce-dbb6c91a62b2" />


- The sql injection is weird because I couldn't read from columns and tables but I could call default functions like `sqlite_version()` but I was able to read the flag by matching for data(sender column) that starts with `l`.I was trying to match `LazyTitan` because that user seems to hold the flag for the chatbot challenges.The main idea is that the statement only allow you to read columsn only from that table.We can do that with the `substr()` sqlite3 function and then match characters with `LIKE`.Matching sender `LazyTitan` seems to spit out the flag but since we specified postion `substr(sender,1,1)`, we'll match only `L`.Payload-:

```json
{"conversation_id":"e3354de4-b91b-4e9f-8e10-f0157210bfed' or substr(sender,1,1) like '%L%'--+"}
```

- Flag-:`CSCTF{H4ck3d_V14_SqL_1nj3ct10n}`

<img width="1535" height="623" alt="image" src="https://github.com/user-attachments/assets/a14a603c-6170-4297-869a-096c4cb2d434" />

---------------

### Extension

---------------

<img width="507" height="910" alt="image" src="https://github.com/user-attachments/assets/c9054004-39e2-4f26-b06f-0ad7666c361b" />

---------------

- I don't get the idea behind this challenge but I guess it is browser escape.Maybe unintended or intended, this was how I solved it.We got presented with a vnc browser to solve the challenge.I couldn't load the urls but I noticed that I could read the files with the `file:///` protocol in the first tabs and not other tabs.

<img width="646" height="278" alt="image" src="https://github.com/user-attachments/assets/bbcf14c3-0108-40ac-aaf0-69dd5708aef7" />

- Now how can we spot important files to read, I noticed an `importing bookmarks` functinality that allows us to import bookmarks on firefox.I can use this to spot important files to read.

<img width="1376" height="727" alt="image" src="https://github.com/user-attachments/assets/1427a18e-d6e1-4a8b-8945-a4c7a255527c" />

- Then, click on other `other files`

<img width="1291" height="877" alt="image" src="https://github.com/user-attachments/assets/dcaf2ffe-281c-455b-bfc6-667433e6aad1" />

- I was able to get other important files to read.

<img width="1121" height="267" alt="image" src="https://github.com/user-attachments/assets/e386fba8-9915-4997-8577-43fa19ae9f79" />

- The flag is located in `/app/embed_flag.py`-:

<img width="850" height="478" alt="image" src="https://github.com/user-attachments/assets/b292c6ba-992f-41ca-9a69-426645ba8d10" />

- Flag-:`CSCTF{1ntricat3_ext3nsi0n_wIth_m@LiciOu$_iNt3nt}`

--------------------







