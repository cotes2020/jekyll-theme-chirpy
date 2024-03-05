require 'time'
require 'socket'

module Yell #:nodoc:

  # Yell::Event.new( :info, 'Hello World', { :scope => 'Application' } )
  # #=> Hello World scope: Application
  class Event
    # regex to fetch caller attributes
    CallerRegexp = /^(.+?):(\d+)(?::in `(.+)')?/

    # jruby and rubinius seem to have a different caller
    CallerIndex = defined?(RUBY_ENGINE) && ["rbx", "jruby"].include?(RUBY_ENGINE) ? 1 : 2


    class Options
      include Comparable

      attr_reader :severity
      attr_reader :caller_offset

      def initialize( severity, caller_offset )
        @severity = severity
        @caller_offset = caller_offset
      end

      def <=>( other )
        @severity <=> other
      end

      alias :to_i :severity
      alias :to_int :severity
    end

    # Prefetch those values (no need to do that on every new instance)
    @@hostname  = Socket.gethostname rescue nil
    @@progname  = $0

    # Accessor to the log level
    attr_reader :level

    # Accessor to the log message
    attr_reader :messages

    # Accessor to the time the log event occured
    attr_reader :time

    # Accessor to the logger's name
    attr_reader :name


    def initialize( logger, options, *messages)
      @time = Time.now
      @name = logger.name

      extract!(options)

      @messages = messages

      @caller = logger.trace.at?(level) ? caller[caller_index].to_s : ''
      @file = nil
      @line = nil
      @method = nil

      @pid = nil
    end

    # Accessor to the hostname
    def hostname
      @@hostname
    end

    # Accessor to the progname
    def progname
      @@progname
    end

    # Accessor to the PID
    def pid
      Process.pid
    end

    # Accessor to the thread's id
    def thread_id
      Thread.current.object_id
    end

    # Accessor to filename the log event occured
    def file
      @file || (backtrace!; @file)
    end

    # Accessor to the line the log event occured
    def line
      @line || (backtrace!; @line)
    end

    # Accessor to the method the log event occured
    def method
      @method || (backtrace!; @method)
    end


    private

    def extract!( options )
      if options.is_a?(Yell::Event::Options)
        @level = options.severity
        @caller_offset = options.caller_offset
      else
        @level = options
        @caller_offset = 0
      end
    end

    def caller_index
      CallerIndex + @caller_offset
    end

    def backtrace!
      if m = CallerRegexp.match(@caller)
        @file, @line, @method = m[1..-1]
      else
        @file, @line, @method = ['', '', '']
      end
    end

  end
end

