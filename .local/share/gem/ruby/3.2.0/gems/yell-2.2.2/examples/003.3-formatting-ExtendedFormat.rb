# encoding: utf-8

require_relative '../lib/yell'

puts <<-EOS

# The extended formatting string looks like: %d [%5L] %p %h : %m.

logger = Yell.new STDOUT, format: Yell::ExtendedFormat
logger.info "Hello World!"
#=> "2012-02-29T09:30:00+01:00 [ INFO] 65784 localhost : Hello World!"
#    ^                          ^      ^     ^           ^
#    ISO8601 Timestamp          Level  Pid   Hostname    Message


EOS

puts "=== actual example ==="
logger = Yell.new STDOUT, format: Yell::ExtendedFormat
logger.info "Hello World!"

