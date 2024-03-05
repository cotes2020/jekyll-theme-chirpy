# encoding: utf-8

require_relative '../lib/yell'

puts <<-EOS

You may colorize the log output on your io-based loggers loke so:

logger = Yell.new STDOUT, colors: true

Yell::Severities.each do |level|
  logger.send level.downcase, level
end

EOS

puts "=== actual example ==="
logger = Yell.new STDOUT, colors: true

Yell::Severities.each do |level|
  logger.send level.downcase, level
end

