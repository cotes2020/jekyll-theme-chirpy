# encoding: utf-8

require_relative '../lib/yell'

puts <<-EOS

# The basic formating string looks like: %l, %d: %m.

logger = Yell.new STDOUT, format: Yell::BasicFormat
logger.info "Hello World!"
#=> "I, 2012-02-29T09:30:00+01:00 : Hello World!"
#    ^  ^                          ^
#    ^  ISO8601 Timestamp          Message
#    Level (short)


EOS

puts "=== actual example ==="
logger = Yell.new STDOUT, format: Yell::BasicFormat
logger.info "Hello World!"

