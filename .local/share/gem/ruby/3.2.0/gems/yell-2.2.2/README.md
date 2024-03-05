# Yell [![Gem Version](https://badge.fury.io/rb/yell.svg)](http://badge.fury.io/rb/yell) [![Build Status](https://travis-ci.org/rudionrails/yell.svg?branch=master)](https://travis-ci.org/rudionrails/yell) [![Code Climate](https://codeclimate.com/github/rudionrails/yell.svg)](https://codeclimate.com/github/rudionrails/yell) [![Coverage Status](https://coveralls.io/repos/rudionrails/yell/badge.svg?branch=master)](https://coveralls.io/r/rudionrails/yell)


**Yell - Your Extensible Logging Library** is a comprehensive logging replacement for Ruby.

Yell works and its test suite currently runs on:

- ruby-head, 2.3.1, 2.2.5, 2.2.2, 2.1.0, 2.0.0 
- jruby-head, jruby-9.1.0.0, jruby-9.0.0.0

If you want to conveniently use Yell with Rails, then head over to [yell-rails](https://github.com/rudionrails/yell-rails). You'll find all the documentation in this repository, though.


## Installation

System wide:

```console
gem install yell
```

Or in your Gemfile:

```ruby
gem "yell"
```


## Usage

On the basics, you can use Yell just like any other logging library with a more 
sophisticated message formatter.

```ruby
logger = Yell.new STDOUT

logger.info "Hello World"
#=> "2012-02-29T09:30:00+01:00 [ INFO] 65784 : Hello World"
#    ^                         ^       ^       ^
#    ISO8601 Timestamp         Level   Pid     Message
```

The strength of Yell, however, comes when using multiple adapters. The already built-in 
ones are IO-based and require no further configuration. Also, there are additional ones 
available as separate gems. Please consult the [wiki](https://github.com/rudionrails/yell/wiki) 
on that - they are listed there.

The standard adapters are:

`:stdout` : Messages will be written to STDOUT  
`:stderr` : Messages will be written to STDERR  
`:file` : Messages will be written to a file  
`:datefile` : Messages will be written to a timestamped file  


Here are some short examples on how to combine them:

##### Example: Notice messages go into `STDOUT` and error messages into `STDERR`

```ruby
logger = Yell.new do |l|
  l.adapter STDOUT, level: [:debug, :info, :warn]
  l.adapter STDERR, level: [:error, :fatal]
end
```

##### Example: Typical production Logger

We setup a logger that starts passing messages at the `:info` level. Severities 
below `:error` go into the 'production.log', whereas anything higher is written 
into the 'error.log'.

```ruby
logger = Yell.new do |l|
  l.level = 'gte.info' # will only pass :info and above to the adapters

  l.adapter :datefile, 'production.log', level: 'lte.warn' # anything lower or equal to :warn
  l.adapter :datefile, 'error.log', level: 'gte.error' # anything greater or equal to :error
end
```

##### Example: Typical production Logger for Heroku

When deploying to Heroku, the "rails_log_stdout" gem gets injected to your Rails project.
Yell does not need that when properly configured (see [yell-rails](https://github.com/rudionrails/yell-rails)
for a more convenient integration with Rails).

```ruby
logger = Yell.new do |l|
  l.level = 'gte.info'

  l.adapter :stdout, level: 'lte.warn'
  l.adapter :stderr, level: 'gte.error'
end
```

### But I'm used to Log4r and I don't want to move on

One of the really nice features of Log4r is its repository. The following example is 
taken from the official Log4r [documentation](http://log4r.rubyforge.org/manual.html#outofbox).

```ruby
require 'log4r'
include Log4r

# create a logger named 'mylog' that logs to stdout
mylog = Logger.new 'mylog'
mylog.outputters = Outputter.stdout

# later in the code, you can get the logger back
Logger['mylog']
```

With Yell you can do the same thing with less:

```ruby
require 'yell'

# create a logger named 'mylog' that logs to stdout
Yell.new :stdout, name: 'mylog'

# later in the code, you can get the logger back
Yell['mylog']
```

There is no need to define outputters separately and you don't have to taint 
you global namespace with Yell's subclasses.

### Adding a logger to an existing class

Yell comes with a simple module: +Yell::Loggable+. Simply include this in a class and 
you are good to go.

```ruby
# Before you can use it, you will need to define a logger and 
# provide it with the `:name` of your class.
Yell.new :stdout, name: 'Foo'

class Foo
  include Yell::Loggable
end

# Now you can log
Foo.logger.info "Hello World"
Foo.new.logger.info "Hello World"
```

It even works with class inheritance:

```ruby
# Given the above example, we inherit from Foo
class Bar < Foo
end

# The logger will fallback to the Foo superclass
Bar.logger.info "Hello World"
Bar.new.logger.info "Hello World"
```

### Adding a logger to all classes at once (global logger)

Derived from the example above, simply do the following.

```ruby
# Define a logger and pass `Object` as name. Internally, Yell adds this
# logger to the repository where you can access it later on.
Yell.new :stdout, name: Object

# Enable logging for the class that (almost) every Ruby class inherits from
Object.send :include, Yell::Loggable

# now you are good to go... from wherever you are
logger.info "Hello from anything"
Integer.logger.info "Hello from Integer"
```

### Suppress log messages with silencers

In case you woul like to suppress certain log messages, you may define
silencers with Yell. Use this to get control of a noisy log environment. For
instance, you can suppress logging messages that contain secure information or
more simply, to skip information about serving your Rails assets. Provide a
string or a regular expression of the message patterns you would like to exclude.

```ruby
logger = Yell.new do |l|
  l.silence /^Started GET "\/assets/
  l.silence /^Served asset/
end

logger.debug 'Started GET "/assets/logo.png" for 127.0.0.1 at 2013-06-20 10:18:38 +0200'
logger.debug 'Served asset /logo.png - 304 Not Modified (0ms)'
```

### Alter log messages with modifiers


## Further Readings

[How To: Setting The Log Level](https://github.com/rudionrails/yell/wiki/101-setting-the-log-level)  
[How To: Formatting Log Messages](https://github.com/rudionrails/yell/wiki/101-formatting-log-messages)  
[How To: Using Adapters](https://github.com/rudionrails/yell/wiki/101-using-adapters)  
[How To: The Datefile Adapter](https://github.com/rudionrails/yell/wiki/101-the-datefile-adapter)  
[How To: Different Adapters for Different Log Levels](https://github.com/rudionrails/yell/wiki/101-different-adapters-for-different-log-levels)  


### Additional Adapters
[Syslog](https://github.com/rudionrails/yell/wiki/additional-adapters-syslog)  
[syslog-sd](https://github.com/raymond-wells/yell-adapters-syslogsd)  
[Graylog2 (GELF)](https://github.com/rudionrails/yell/wiki/additional-adapters-gelf)  
[Fluentd](https://github.com/red5studios/yell-adapters-fluentd)  


### Development

[How To: Writing Your Own Adapter](https://github.com/rudionrails/yell/wiki/Writing-your-own-adapter)  

You can find further examples and additional adapters in the [wiki](https://github.com/rudionrails/yell/wiki).
or have a look into the examples folder.


Copyright &copy; 2011-current Rudolf Schmidt, released under the MIT license

