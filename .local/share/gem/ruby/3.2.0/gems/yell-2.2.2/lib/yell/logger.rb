require 'pathname'

module Yell #:nodoc:
  # The +Yell::Logger+ is your entrypoint. Anything onwards is derived from here.
  #
  # A +Yell::Logger+ instance holds all your adapters and sends the log events
  # to them if applicable. There are multiple ways of how to create a new logger.
  class Logger
    include Yell::Helpers::Base
    include Yell::Helpers::Level
    include Yell::Helpers::Formatter
    include Yell::Helpers::Adapter
    include Yell::Helpers::Tracer
    include Yell::Helpers::Silencer

    # The name of the logger instance
    attr_reader :name

    # Initialize a new Logger
    #
    # @example A standard file logger
    #   Yell::Logger.new 'development.log'
    #
    # @example A standard datefile logger
    #   Yell::Logger.new :datefile
    #   Yell::Logger.new :datefile, 'development.log'
    #
    # @example Setting the log level
    #   Yell::Logger.new level: :warn
    #
    #   Yell::Logger.new do |l|
    #     l.level = :warn
    #   end
    #
    # @example Combined settings
    #   Yell::Logger.new 'development.log', level: :warn
    #
    #   Yell::Logger.new :datefile, 'development.log' do |l|
    #     l.level = :info
    #   end
    def initialize( *args, &block )
      # extract options
      @options = args.last.is_a?(Hash) ? args.pop : {}

      # check if filename was given as argument and put it into the @options
      if [String, Pathname].include?(args.last.class)
        @options[:filename] = args.pop unless @options[:filename]
      end

      reset!

      # FIXME: :format is deprecated in future versions --R
      self.formatter = Yell.__fetch__(@options, :format, :formatter)
      self.level = Yell.__fetch__(@options, :level, :default => 0)
      self.name = Yell.__fetch__(@options, :name)
      self.trace = Yell.__fetch__(@options, :trace, default: :error)

      # silencer
      self.silence(*Yell.__fetch__(@options, :silence, default: []))

      # adapters may be passed in the options
      extract!(*Yell.__fetch__(@options, :adapters, default: []))

      # extract adapter
      self.adapter(args.pop) if args.any?

      # eval the given block
      block.arity > 0 ? block.call(self) : instance_eval(&block) if block_given?

      # default adapter when none defined
      self.adapter(:file) if adapters.empty?
    end


    # Set the name of a logger. When providing a name, the logger will
    # automatically be added to the Yell::Repository.
    #
    # @return [String] The logger's name
    def name=( val )
      Yell::Repository[val] = self if val
      @name = val.nil? ? "<#{self.class.name}##{object_id}>": val

      @name
    end

    # Somewhat backwards compatible method (not fully though)
    def add( options, *messages, &block )
      return false unless level.at?(options)

      messages = messages
      messages << block.call unless block.nil?
      messages = silencer.call(*messages)
      return false if messages.empty?

      event = Yell::Event.new(self, options, *messages)
      write(event)
    end

    # Creates instance methods for every log level:
    #   `debug` and `debug?`
    #   `info` and `info?`
    #   `warn` and `warn?`
    #   `error` and `error?`
    #   `unknown` and `unknown?`
    Yell::Severities.each_with_index do |s, index|
      name = s.downcase

      class_eval <<-EOS, __FILE__, __LINE__ + index
        def #{name}?; level.at?(#{index}); end            # def info?; level.at?(1); end
                                                          #
        def #{name}( *m, &b )                             # def info( *m, &b )
          options = Yell::Event::Options.new(#{index}, 1)
          add(options, *m, &b)                            #   add(Yell::Event::Options.new(1, 1), *m, &b)
        end                                               # end
      EOS
    end

    # Get a pretty string representation of the logger.
    def inspect
      inspection = inspectables.map { |m| "#{m}: #{send(m).inspect}" }
      "#<#{self.class.name} #{inspection * ', '}>"
    end

    # @private
    def close
      adapters.close
    end

    # @private
    def write( event )
      adapters.write(event)
    end

    private

    # The :adapters key may be passed to the options hash. It may appear in
    # multiple variations:
    #
    # @example
    #   extract!(:stdout, :stderr)
    #
    # @example
    #   extract!(stdout: {level: :info}, stderr: {level: :error})
    def extract!( *list )
      list.each do |a|
        if a.is_a?(Hash)
          a.each { |t, o| adapter(t, o) }
        else
          adapter(a)
        end
      end
    end

    # Get an array of inspected attributes for the adapter.
    def inspectables
      [:name] | super
    end
  end
end

