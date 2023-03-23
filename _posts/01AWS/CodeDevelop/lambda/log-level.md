


[toc]

- ref
  - [Logging Levels: What They Are and How They Help You](https://www.scalyr.com/blog/logging-levels/)
  - [Logging in Python](https://realpython.com/python-logging/)
  - [Logger](https://awslabs.github.io/aws-lambda-powertools-python/core/logger/)

---

# logging in python

---

## The Logging Module

use `logger`
- to log messages that you want to see.
- By default, there are 5 standard levels indicating the severity of events.
  - `CRITICAL` (highest severity)
  - `ERROR`
  - `WARNING`
  - `INFO`
  - `DEBUG`



debug > info > warning > error


```py
import logging
logging.debug('This is a debug message')
logging.info('This is an info message')
logging.warning('This is a warning message')
logging.error('This is an error message')
logging.critical('This is a critical message')
# The output:
# WARNING:root:This is a warning message
# ERROR:root:This is an error message
# CRITICAL:root:This is a critical message
```

<font color=red> level::name::message </font>

- The output shows the severity level before each message along with root, the name the logging module gives to its default logger.
- default output format that can be configured to include things like timestamp, line number, and other details.

the `debug()` and `info()` messages didn’t get logged.
- by default, the logging module logs the messages with a severity level of WARNING or above.
- configuring the logging module to log events of all levels
- can also define your own severity levels by changing configurations,
- not recommended as it can cause confusion with logs of some third-party libraries that you might be using.


## Basic Configurations
- use the `basicConfig(**kwargs)` method to configure the logging
  - it can configure the `root logger` works only if the root logger has not been configured before.
  - this function can only be called once.

  - commonly used parameters for `basicConfig()` are the following:
    - `level`:
      - The root logger will be set to the specified severity level.
    - `filename`: This specifies the file.
    - `filemode`: If filename is given, the file is opened in this mode. The default is a, which means append.
    - `format`: the format of the log message.
  - using the level parameter, set what level of log messages to record.
  - passing the constants available in the class, and this would enable all logging calls at or above that level to be logged.

```py
import logging

# All events at or above DEBUG level will get logged.
logging.basicConfig(level=logging.DEBUG)
logging.debug('This will get logged')
# DEBUG:root:This will get logged

# to log to a file not console, filename and filemode can be used
logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logging.warning('This will get logged to a file')
# format:
# root - ERROR - This will get logged to a file
```

---

### Formatting the Output

- pass variable that can be represented as a string from your program as a message to your logs
- some basic elements that are already a part of the `LogRecord` and can be easily added to the output format.
  - `LogRecord` attributes
  - The entire list of available attributes can be found [here](https://docs.python.org/3/library/logging.html#logrecord-attributes).

```py
import logging

# to log the process ID along with the level and message
logging.basicConfig(format='%(process)d-%(levelname)s-%(message)s')
logging.warning('This is a Warning')
# 18472-WARNING-This is a Warning


# add the date and time info
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logging.info('Admin logged in')
# 2018-07-11 20:12:06,288 - Admin logged in


# `%(asctime)s` adds the time of creation of the `LogRecord`.
# - The format can be changed using the `datefmt` attribute
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.warning('Admin logged out')
12-Jul-18 20:53:19 - Admin logged out
```

---

### Logging Variable Data

to include dynamic information from your application in the logs.
- format a string with variable data in a separate line and pass it to the log method.
- using a format string for the message and appending the variable data as arguments.
- The arguments passed to the method would be included as variable data in the message.


```py
import logging
name = 'John'
logging.error('%s raised an error', name)
# ERROR:root:John raised an error

logging.error(f'{name} raised an error')
# ERROR:root:John raised an error
```

---

### Capturing Stack Traces
1. `logging.error()`
   - The logging module allows you to capture the full stack traces in an application.
   - Exception information can be captured if the `exc_info` parameter is passed as `True`

```py
import logging
a = 5
b = 0
try:
  c = a / b
except Exception as e:
  logging.error("Exception occurred", exc_info=True)
# ERROR:root:Exception occurred
# Traceback (most recent call last):
#  File "exceptions.py", line 6, in <module>
#  c = a / b
# ZeroDivisionError: division by zero
# [Finished in 0.2s]
```


2. `logging.exception()` method
   - <font color=blue> logging from an exception handler </font>
   - logs a message with level `ERROR` and adds exception information to the message.
     - `logging.exception()` would show a log at the level of `ERROR`.
     - If you don’t want that
       - call any of the other logging methods from `debug()` to `critical()`
       - and pass the `exc_info` parameter as `True`.
   - calling `logging.exception()` is like calling `logging.error(exc_info=True)`.
   - But since this method always dumps exception information, it should only be called from an exception handler.


```py
import logging
a = 5
b = 0
try:
  c = a / b
except Exception as e:
  logging.exception("Exception occurred")
# ERROR:root:Exception occurred
# Traceback (most recent call last):
#  File "exceptions.py", line 6, in <module>
#    c = a / b
# ZeroDivisionError: division by zero
# [Finished in 0.2s]
```

---

## customer `logger`

---

### Classes and Functions in the module.


the default logger named `root`
- used by the logging module whenever its functions are called directly like this: `logging.debug()`.
- define your own logger by creating an object of the `Logger` class, especially if your application has multiple modules.


The most commonly used classes defined in the logging module

- **`Logger`:**
  - This is the class
  - objects will be used in the application code directly to call the functions.

- **`LogRecord`:**
  - Loggers automatically create `LogRecord` objects that have all the information related to the event being logged,
  - like the name of the logger, the function, the line number, the message, and more.

- **`Handler`:**
  - Handlers send the `LogRecord` to the required output destination,
    - like the console or a file.
  - `Handler` is a base for subclasses like `StreamHandler`, `FileHandler`, `SMTPHandler`, `HTTPHandler`, and more.
  - These subclasses send the logging outputs to corresponding destinations,
    - like `sys.stdout` or a disk file.

- **`Formatter`:**
  - This is where you specify the format of the output by specifying a string format that lists out the attributes that the output should contain.


---

### the `Logger` class
- instantiated using the module-level function `logging.getLogger(name)`.
- Multiple calls to `getLogger()` with the same `name` will return a reference to the same `Logger` object, which saves us from passing the logger objects to every part where it’s needed.


```py
import logging

logger = logging.getLogger('example_logger')
logger.warning('This is a warning')
# This is a warning
```

- This creates a custom logger named `example_logger`,
- unlike the root logger
  - the name of a custom logger is not part of the default output format and has to be added to the configuration.
    - Configuring it to a format to show the name of the logger would give an output like this:
    - `WARNING:example_logger:This is a warning`
  - custom logger can’t be configured using `basicConfig()`
    - have to configure it using `Handlers` and `Formatters`:

> “It is recommended that we use module-level loggers by passing `__name__` as the name parameter to `getLogger()` to create a logger object as the name of the logger itself would tell us from where the events are being logged. `__name__` is a special built-in variable in Python which evaluates to the name of the current module.”


```py
# app.py
from aws_lambda_powertools import Logger
logger = Logger() # Sets service via env var
# OR logger = Logger(service="example")
```


```yaml
Resources:
  HelloWorldFunction:
    Type: AWS::Serverless::Function
    Properties:
      Runtime: python3.8
      Environment:
        Variables:
          LOG_LEVEL: INFO
          POWERTOOLS_SERVICE_NAME: example
```






---

### Using Handlers

- to configure your own loggers and send the logs to multiple places when they are generated.
  - Handlers send the log messages to configured destinations like
    - the standard output stream
    - or a file
    - or over HTTP
    - or to your email via SMTP.
- A logger can have more than one handler
  - you can saved to a log file and also send it over email.
- can also set the severity level in handlers.
  - useful to set multiple handlers for the same logger but want different severity levels for each of them.
  - For example
  - logs with level `WARNING` and above to be logged to the console,
  - but everything with level `ERROR` and above should also be saved to a file.


1. creating a `LogRecord`
   - it holds all the information of the event
   - and passing it to all the Handlers it has: `c_handler` and `f_handler`.
2. `c_handler`
   - a `StreamHandler` with level `WARNING`
   - takes the info from the `LogRecord` to generate an output in the format specified
   - and prints it to the console.
3. `f_handler`
   - a `FileHandler` with level `ERROR`
   - it ignores this `LogRecord` as its level is `WARNING`.

4. When `logger.error()` is called
   - `c_handler` behaves exactly as before,
   - and `f_handler` gets a `LogRecord` at the level of `ERROR`, so it proceeds to generate an output just like `c_handler`, but instead of printing it to console, it writes it to the specified file in this format:

```py
# logging_example.py
import logging

# Create a custom logger
logger = logging.getLogger(__name__)

# Create handlers
c_handler = logging.StreamHandler()
c_handler.setLevel(logging.WARNING)
f_handler = logging.FileHandler('file.log')
f_handler.setLevel(logging.ERROR)
# Create formatters and add it to handlers
c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)

logger.warning('This is a warning')
logger.error('This is an error')

# __main__ - WARNING - This is a warning
# __main__ - ERROR - This is an error
# 2018-08-03 16:12:21,723 - __main__ - ERROR - This is an error
```

The name of the logger corresponding to the `__name__` variable is logged as `__main__`, which is the name Python assigns to the module where execution starts. If this file is imported by some other module, then the `__name__` variable would correspond to its name _logging\_example_. Here’s how it would look:

```py
# run.py
import logging_example
# logging_example - WARNING - This is a warning
# logging_example - ERROR - This is an error
```

---


### Other Configuration Methods

configure logging
- using the <font color=blue> module and class functions </font>
- or creating a config file or a [dictionary](https://realpython.com/python-dicts/) and loading it using `fileConfig()` or `dictConfig()` respectively.
- useful in case to change your logging configuration in a running application.

Here’s an example file configuration:

```yaml
[loggers]
keys=root,sampleLogger

[logger_root]
level=DEBUG
handlers=consoleHandler

[logger_sampleLogger]
level=DEBUG
handlers=consoleHandler
qualname=sampleLogger
propagate=0


[handlers]
keys=consoleHandler

[handler_consoleHandler]
class=StreamHandler
level=DEBUG
formatter=sampleFormatter
args=(sys.stdout,)

[formatters]
keys=sampleFormatter

[formatter_sampleFormatter]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

In the above file
- two loggers, one handler, and one formatter.
- After their names are defined, they are configured by adding the words logger, handler, and formatter before their names separated by an underscore.

To load this config file
- use `fileConfig()`:

```py
import logging
import logging.config

logging.config.fileConfig(fname='file.conf', disable_existing_loggers=False)

# Get the logger specified in the file
logger = logging.getLogger(__name__)

logger.debug('This is a debug message')

# 2018-07-13 13:57:45,467 - __main__ - DEBUG - This is a debug message
```

- The path of the config file is passed as a parameter to the `fileConfig()` method
- the `disable_existing_loggers` parameter is used to keep or disable the loggers that are present when the function is called. It defaults to `True` if not mentioned.

Here’s the same configuration in a YAML format for the dictionary approach:

```yaml
version: 1
formatters:
  simple:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
handlers:
  console:
    class: logging.StreamHandler
    level: DEBUG
    formatter: simple
    stream: ext://sys.stdout
loggers:
  sampleLogger:
    level: DEBUG
    handlers: [console]
    propagate: no
root:
  level: DEBUG
  handlers: [console]
```

Here’s an example that shows how to load config from a `yaml` file:

```py
import logging
import logging.config
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)

logger = logging.getLogger(__name__)

logger.debug('This is a debug message')
# 2018-07-13 14:05:03,766 - __main__ - DEBUG - This is a debug message
```


---

## lambda logger

### Standard structured keys

Logger will include the following keys to structured logging, by default:

| Key           | Type | Example              | Description                                                       |
| ------------- | ---- |
| timestamp     | str  | "2020-05-24"         | Timestamp of log statement                                        |
| level         | str  | "INFO"               | Logging level                                                     |
| location      | str  | "collect.handler:1"  | Source code location where statement was executed                 |
| service       | str  | "payment"            | Service name defined. "service_undefined" will be used if unknown |
| sampling_rate | int  | 0.1                  | Debug logging sampling rate in percentage e.g. 10% in this case   |
| message       | any  | "Collecting payment" | Log statement value. Unserializable JSON casted to string         |
| xray_trace_id | str  | "1-5759e988-bd83"    | X-Ray Trace ID when Lambda function has enabled Tracing           |




---

### Capturing Lambda context info
enrich structured logs with key Lambda context information via inject_lambda_context.


```py
from aws_lambda_powertools import Logger
logger = Logger()

@logger.inject_lambda_context
def handler(event, context):
 logger.info("Collecting payment")
 ...
 # You can log entire objects too
 logger.info({
    "operation": "collect_payment",
    "charge_id": event['charge_id']
 })
 ...

{
  "timestamp": "2020-05-24 18:17:33,774",
  "level": "INFO",
  "location": "collect.handler:1",
  "service": "payment",
  "lambda_function_name": "test",
  "lambda_function_memory_size": 128,
  "lambda_function_arn": "arn:aws:lambda:eu-west-1:12345678910:function:test",
  "lambda_request_id": "52fdfc07-2182-154f-163f-5f0f9a621d72",
  "cold_start": true,
  "sampling_rate": 0.0,
  "message": "Collecting payment"
},
{
  "timestamp": "2020-05-24 18:17:33,774",
  "level": "INFO",
  "location": "collect.handler:15",
  "service": "payment",
  "lambda_function_name": "test",
  "lambda_function_memory_size": 128,
  "lambda_function_arn": "arn:aws:lambda:eu-west-1:12345678910:function:test",
  "lambda_request_id": "52fdfc07-2182-154f-163f-5f0f9a621d72",
  "cold_start": true,
  "sampling_rate": 0.0,
  "message": {
    "operation": "collect_payment",
    "charge_id": "ch_AZFlk2345C0"
  }
}
```

---

### Appending additional keys¶

- append additional keys using either mechanism:
- Persist new keys across all future log messages via structure_logs method
- Add additional keys on a per log message basis via extra parameter


```py
from aws_lambda_powertools import Logger
logger = Logger()


# 1. `structure_logs` method
#    - append your own keys to your existing Logger via `structure_logs(append=True, **kwargs)` method.
#    - Omitting append=True will reset the existing structured logs to standard keys + keys provided as arguments

def handler(event, context):
 order_id = event.get("order_id")
 logger.structure_logs(append=True, order_id=order_id)
 logger.info("Collecting payment")

{
  "timestamp": "2020-05-24 18:17:33,774",
  "level": "INFO",
  "location": "collect.handler:1",
  "service": "payment",
  "sampling_rate": 0.0,
  "order_id": "order_id_value",
  "message": "Collecting payment"
}



# 2. extra parameter¶
#    - Extra parameter is available for all log levels' methods (implemented in the standard logging library)
#      - e.g. logger.info, logger.warning.
#    - It accepts any dictionary, and all keyword arguments will be added as part of the root structure of the logs for that log statement.
#    - Any keyword argument added using extra will not be persisted for subsequent messages.

logger = Logger(service="payment")
fields = { "request_id": "1123" }
logger.info("Hello", extra=fields)

{
  "timestamp": "2021-01-12 14:08:12,357",
  "level": "INFO",
  "location": "collect.handler:1",
  "service": "payment",
  "sampling_rate": 0.0,
  "request_id": "1123",
  "message": "Collecting payment"
}

```


---

## Advanced

### Reusing Logger across your code
- Logger supports inheritance via child parameter.
- allows you to create multiple Loggers across your code base, and propagate changes such as new keys to all Loggers.

```py
# collect.py
import shared # Creates a child logger named "payment.shared"
from aws_lambda_powertools import Logger

logger = Logger() # POWERTOOLS_SERVICE_NAME: "payment"

def handler(event, context):
    shared.inject_payment_id(event)


#shared.py
from aws_lambda_powertools import Logger

logger = Logger(child=True) # POWERTOOLS_SERVICE_NAME: "payment"

def inject_payment_id(event):
    logger.structure_logs(append=True, payment_id=event.get("payment_id"))
```

1. Logger will create a **parent logger** named `payment` and a **child logger** named `payment.shared`.
2. Changes in either parent or child logger will be propagated bi-directionally.
3. Child loggers will be named after the following convention `{service}.{filename}`
4. If you forget to use child param but the service name is the same of the parent, we will return the existing parent Logger instead.
















---

# Application logging


---

# Logging Levels

Application logging is one of the most important things to facilitating production support.
- log files serve as a sort of archaeological record of what on earth your codebase did in production. 
- Each entry in a log file has important information, including a time stamp, contextual information, logging level, and a message.



> capture every last detail you can because this might prove useful during troubleshooting or auditing your system. 
> all logging consumes resources. eat up disk space, overload people reading the logs, and even start to slow down your production code if you go overboard.



logging levels
- categorizing the entries in your log file.
- the logging level lets you separate the information
- distinction helps in two ways. 
  - filter log files during search.
  - control the amount of information that you log.
- logging requires either a balance to get both the proverbial signal _and_ the noise.
  - Logging levels work this way. 

---


## Common Logging Levels


`debug > info >   warning > error`

### FATAL 灾难性的

Fatal
- represents truly catastrophic situations, as far as your application is concerned. 
  - application is about to abort to prevent some kind of corruption or serious problem
- This entry in the log should probably result in someone getting a 3 AM phone call.

### ERROR

error
- serious issue and represents the failure of something important going on in your application. 
- Unlike FATAL, the application itself isn’t going down the tubes. 
  - like dropped database connections or the inability to access a file or service. 
- This will require someone’s attention probably sooner than later, but the application can limp along.


### WARN 假设的,假定的;有待证实的

WARN
- _might_ have a problem and that have detected an unusual situation. 
  - invoke a service and it failed a couple of times before connecting on an automatic retry. 
- unexpected and unusual, but no real harm done,
- and not known whether the issue will persist or recur. 
- Someone should investigate warnings.

### INFO

INFO
- normal application behavior and milestones. 
- won’t care too much about these entries during normal operations, but they provide the skeleton of what happened. 
- A service started or stopped.  You added a new user to the database.  That sort of thing.

### DEBUG

DEBUG
- include more granular, diagnostic information. 
- furnishing more information than you’d want in normal production situations. 
- providing detailed diagnostic information for fellow developers, sysadmins, etc.

### TRACE

TRACE
- fine-grained information—finer even than DEBUG. 
- capture every detail possibly can about the application’s behavior. 
- This is likely to swamp your resources in production and is seriously diagnostic.

### ALL

Log absolutely everything, including any custom logging levels that someone has defined.

### OFF

Don’t log anything at all.


---


## How This Works

two participating parties in logging:

- The logging framework, at runtime, has a configured log level.
- The application code makes logging requests.

If the framework has a given log level enabled, then all requests at that level or higher priority wind up in the log file.  Everything else is denied.  So consider the following pseudo-code:

```java
void DoStuffWithInts(int x, int y) {
    log.trace(x);
    log.error(y);
}
```

- log level set to `ALL` or `TRACE`, you would see both integers in the log file. 
- log level set to `WARN`, then we would only see _y_. 
- log level set to `FATAL`, see nothing.
