# frozen_string_literal: true

require "yell"
require "rainbow"

module HTMLProofer
  class Log
    include Yell::Loggable

    STDOUT_LEVELS = [:debug, :info, :warn].freeze
    STDERR_LEVELS = [:error, :fatal].freeze

    def initialize(log_level)
      @logger = Yell.new(format: false, \
        name: "HTMLProofer", \
        level: "gte.#{log_level}") do |l|
        l.adapter(:stdout, level: "lte.warn")
        l.adapter(:stderr, level: "gte.error")
      end
    end

    def log(level, message)
      log_with_color(level, message)
    end

    def log_with_color(level, message)
      @logger.send(level, colorize(level, message))
    end

    def colorize(level, message)
      color = case level
      when :debug
        :cyan
      when :info
        :blue
      when :warn
        :yellow
      when :error, :fatal
        :red
      end

      if (STDOUT_LEVELS.include?(level) && $stdout.isatty) || \
          (STDERR_LEVELS.include?(level) && $stderr.isatty)
        Rainbow(message).send(color)
      else
        message
      end
    end

    # dumb override to play nice with Typhoeus/Ethon
    def debug(message = nil)
      log(:debug, message) unless message.nil?
    end
  end
end
