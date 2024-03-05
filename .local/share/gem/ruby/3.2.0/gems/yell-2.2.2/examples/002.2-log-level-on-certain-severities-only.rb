# encoding: utf-8

require_relative '../lib/yell'

puts <<-EOS

# The Yell::Level parser allows you to exactly specify on which levels to log, 
# ignoring all the others. For instance: If we want to only log at the :debug 
# and :warn levels we simply providing an array:
# * %i[] is a built-in for an array of symbols

logger = Yell.new STDOUT, level: %i[debug warn]

%i[debug info warn error fatal].each do |level| 
  logger.send( level, level )
end
#=> "2012-02-29T09:30:00+01:00 [DEBUG] 65784 : debug"
#=> "2012-02-29T09:30:00+01:00 [ WARN] 65784 : warn"


EOS

puts "=== actual example ==="
logger = Yell.new STDOUT, level: %i[debug warn]

%i[debug info warn error fatal].each do |level| 
  logger.send( level, level )
end

