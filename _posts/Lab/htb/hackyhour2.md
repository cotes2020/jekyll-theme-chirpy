

# hacky hour 2

[toc]

---


# 2

```bash
# go to cache.htb 10.10.10.188

view-source:http://cache.htb/jquery/functionality.js
# $(function(){
#     var error_correctPassword = false;
#     var error_username = false;
#     function checkCorrectPassword(){
#         var Password = $("#password").val();
#         if(Password != 'H@v3_fun'){
#             alert("Password didn't Match");
#             error_correctPassword = true;
#         }
#     }
#     function checkCorrectUsername(){
#         var Username = $("#username").val();
#         if(Username != "ash"){
#             alert("Username didn't Match");
#             error_username = true;
#         }
#     }
#     $("#loginform").submit(function(event) {
#         /* Act on the event */
#         error_correctPassword = false;
#          checkCorrectPassword();
#          error_username = false;
#          checkCorrectUsername();
#         if(error_correctPassword == false && error_username ==false){
#             return true;
#         }
#         else{
#             return false;
#         }
#     });
# });




view-source:http://cache.htb/net.html
# <html>
# <head>
#  <body onload="if (document.referrer == '') self.location='login.html';">
# 	<style>
#       body  {
#         background-color: #cccccc;
#       }
#   </style>
# </head>
# <center>
# 	<h1> Welcome Back!</h1>
# 	<img alt="pic" src="4202252.jpg">
# <h1>This page is still underconstruction</h1>
# </center>
#  </body>
# </html>



wfuzz -c -z file./usr/share/wordlists/rockyou.txt cache.htb/FUZZ


# virtual host
# wfuzz -w /path/to/wordlist -H "Host: FUZZ.host.com" --hc 200 --hw 356 -t 100 10.10.10.188

wfuzz -w /usr/share/wordlists/rockyou.txt -H "Host: http://cache.htb" --hc 200 --hw 356 -t 100 10.10.10.188

wfuzz -w /usr/share/wordlists/rockyou.txt -H "http://FUZZ.cache.htb" --hc 200 --hw 356 -t 100 10.10.10.188

wfuzz -w /usr/share/wordlists/rockyou.txt http://FUZZ.htb --hc 200 --hw 356 -t 100 10.10.10.188

wfuzz -w /usr/share/wordlists/rockyou.txt http://FUZZ.htb

wfuzz -w /usr/share/wordlists/rockyou.txt -H "http://FUZZ.cache.htb" -t 100 10.10.10.188

wfuzz -w /usr/share/wordlists/rockyou.txt http://FUZZ.cache.htb/FUZZ




```
