# encoding: utf-8

require_relative '../lib/yell'

puts <<-EOS

# Like many other logging libraries, Yell allows you to define from which level 
# onwards you want to write your log message. 

logger = Yell.new STDOUT, level: :info

logger.debug "This is a :debug message"
#=> nil 

logger.info "This is a :info message"
#=> "2012-02-29T09:30:00+01:00 [ INFO] 65784 : This is a :info message"


EOS

puts "=== actual example ==="
logger = Yell.new STDOUT, level: :info

logger.debug "This is a :debug message"
logger.info "This is a :info message"
