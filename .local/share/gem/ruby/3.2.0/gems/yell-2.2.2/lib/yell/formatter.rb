require 'time'

# TODO: Register custom formats
#
# @example The Yell default fomat
#   Yell::Formatter.register(:default)
#
# @example The Ruby standard logger format
#   Yell::Formatter.register(:stdlogger, "%l, [%d #%p] %5L -- : %m", "%Y-%m-%dT%H:%M:%S.%6N")
#
module Yell #:nodoc:

  # No format on the log message
  #
  # @example
  #   logger = Yell.new STDOUT, format: false
  #   logger.info "Hello World!"
  #   #=> "Hello World!"
  NoFormat = "%m"

  # Default Format
  #
  # @example
  #   logger = Yell.new STDOUT, format: Yell::DefaultFormat
  #   logger.info "Hello World!"
  #   #=> "2012-02-29T09:30:00+01:00 [ INFO] 65784 : Hello World!"
  #   #    ^                         ^       ^       ^
  #   #    ISO8601 Timestamp         Level   Pid     Message
  DefaultFormat = "%d [%5L] %p : %m"

  # Basic Format
  #
  # @example
  #   logger = Yell.new STDOUT, format: Yell::BasicFormat
  #   logger.info "Hello World!"
  #   #=> "I, 2012-02-29T09:30:00+01:00 : Hello World!"
  #   #    ^  ^                          ^
  #   #    ^  ISO8601 Timestamp          Message
  #   #    Level (short)
  BasicFormat = "%l, %d : %m"

  # Extended Format
  #
  # @example
  #   logger = Yell.new STDOUT, format: Yell::ExtendedFormat
  #   logger.info "Hello World!"
  #   #=> "2012-02-29T09:30:00+01:00 [ INFO] 65784 localhost : Hello World!"
  #   #    ^                          ^      ^     ^           ^
  #   #    ISO8601 Timestamp          Level  Pid   Hostname    Message
  ExtendedFormat  = "%d [%5L] %p %h : %m"


  # The +Formatter+ provides a handle to configure your log message style.
  class Formatter

    Table = {
      "m" => "message(event.messages)",    # Message
      "l" => "level(event.level, 1)",      # Level (short), e.g.'I', 'W'
      "L" => "level(event.level)",         # Level, e.g. 'INFO', 'WARN'
      "d" => "date(event.time)",           # ISO8601 Timestamp
      "h" => "event.hostname",             # Hostname
      "p" => "event.pid",                  # PID
      "P" => "event.progname",             # Progname
      "t" => "event.thread_id",            # Thread ID
      "F" => "event.file",                 # Path with filename where the logger was called
      "f" => "File.basename(event.file)",  # Filename where the loger was called
      "M" => "event.method",               # Method name where the logger was called
      "n" => "event.line",                 # Line where the logger was called
      "N" => "event.name"                  # Name of the logger
    }

    # For standard formatted backwards compatibility
    LegacyTable = Hash[ Table.keys.map { |k| [k, 'noop'] } ].merge(
      'm' => 'message(msg)',
      'l' => 'level(event, 1)',
      'L' => 'level(event)',
      'd' => 'date(time)',
      "p" => "$$",
      'P' => 'progname'
    )

    PatternMatcher = /([^%]*)(%\d*)?(#{Table.keys.join('|')})?(.*)/m


    attr_reader :pattern, :date_pattern


    # Initializes a new +Yell::Formatter+.
    #
    # Upon initialization it defines a format method. `format` takes
    # a {Yell::Event} instance as agument in order to apply for desired log
    # message formatting.
    #
    # @example Blank formatter
    #   Formatter.new
    #
    # @example Formatter with a message pattern
    #   Formatter.new("%d [%5L] %p : %m")
    #
    # @example Formatter with a message and date pattern
    #   Formatter.new("%d [%5L] %p : %m", "%D %H:%M:%S.%L")
    #
    # @example Formatter with a message modifier
    #   Formatter.new do |f|
    #     f.modify(Hash) { |h| "Hash: #{h.inspect}" }
    #   end
    def initialize( *args, &block )
      builder = Builder.new(*args, &block)

      @pattern = builder.pattern
      @date_pattern = builder.date_pattern
      @modifier = builder.modifier

      define_date_method!
      define_call_method!
    end

    # Get a pretty string
    def inspect
      "#<#{self.class.name} pattern: #{@pattern.inspect}, date_pattern: #{@date_pattern.inspect}>"
    end


    private

    # Message modifier class to allow different modifiers for different requirements.
    class Modifier
      def initialize
        @repository = {}
      end

      def set( key, &block )
        @repository.merge!(key => block)
      end

      def call( message )
        case
        when mod = @repository[message.class] || @repository[message.class.to_s]
          mod.call(message)
        when message.is_a?(Array)
          message.map { |m| call(m) }.join(" ")
        when message.is_a?(Hash)
          message.map { |k, v| "#{k}: #{v}" }.join(", ")
        when message.is_a?(Exception)
          backtrace = message.backtrace ? "\n\t#{message.backtrace.join("\n\t")}" : ""
          sprintf("%s: %s%s", message.class, message.message, backtrace)
        else
          message
        end
      end
    end

    # Builder class to allow setters that won't be accessible once
    # transferred to the Formatter
    class Builder
      attr_accessor :pattern, :date_pattern
      attr_reader :modifier

      def initialize( pattern = nil, date_pattern = nil, &block )
        @modifier = Modifier.new

        @pattern = case pattern
        when false then Yell::NoFormat
        when nil then Yell::DefaultFormat
        else pattern
        end.dup

        @pattern << "\n" unless @pattern[-1] == ?\n # add newline if not present
        @date_pattern = date_pattern || :iso8601

        block.call(self) if block
      end

      def modify( key, &block )
        modifier.set(key, &block)
      end
    end

    def define_date_method!
      buf = case @date_pattern
      when String then "t.strftime(@date_pattern)"
      when Symbol then respond_to?(@date_pattern, true) ? "#{@date_pattern}(t)" : "t.#{@date_pattern}"
      else t.iso8601
      end

      # define the method
      instance_eval <<-METHOD, __FILE__, __LINE__
        def date(t = Time.now)
          #{buf}
        end
       METHOD
    end

    # define a standard +Logger+ backwards compatible #call method for the formatter
    def define_call_method!
      instance_eval <<-METHOD, __FILE__, __LINE__
        def call(event, time = nil, progname = nil, msg = nil)
          event.is_a?(Yell::Event) ? #{to_sprintf(Table)} : #{to_sprintf(LegacyTable)}
        end
      METHOD
    end

    def to_sprintf( table )
      buff, args, _pattern = "", [], @pattern.dup

      while true
        match = PatternMatcher.match(_pattern)

        buff << match[1] unless match[1].empty?
        break if match[2].nil?

        buff << match[2] + 's'
        args << table[ match[3] ]

        _pattern = match[4]
      end

      %Q{sprintf("#{buff.gsub(/"/, '\"')}", #{args.join(', ')})}
    end

    def level( sev, length = nil )
      severity = case sev
      when Integer then Yell::Severities[sev] || 'ANY'
      else sev
      end

      length.nil? ? severity : severity[0, length]
    end

    def message( messages )
      @modifier.call(messages.is_a?(Array) && messages.size == 1 ? messages.first : messages)
    end

    # do nothing
    def noop
      ''
    end

  end
end

