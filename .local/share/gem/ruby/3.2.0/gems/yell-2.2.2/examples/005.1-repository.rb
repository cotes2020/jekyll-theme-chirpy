# encoding: utf-8

require_relative '../lib/yell'

puts <<-EOS

# You can add a logger to the global repository.
#
# create a logger named 'mylog' that logs to stdout
Yell.new :stdout, name: 'mylog'

# Later in the code, you can get your logger back
Yell['mylog'].info "Hello World!"


EOS

puts "=== actual example ==="
Yell.new :stdout, name: 'mylog'
Yell['mylog'].info "Hello World!"

