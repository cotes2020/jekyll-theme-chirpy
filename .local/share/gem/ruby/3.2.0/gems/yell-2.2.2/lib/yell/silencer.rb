module Yell #:nodoc:

  # The +Yell::Silencer+ is your handly helper for stiping out unwanted log messages.
  class Silencer

    class PresetNotFound < StandardError
      def message; "Could not find a preset for #{super.inspect}"; end
    end

    Presets = {
      :assets => [/\AStarted GET "\/assets/, /\AServed asset/, /\A\s*\z/] # for Rails
    }


    def initialize( *patterns )
      @patterns = patterns.dup
    end

    # Add one or more patterns to the silencer
    #
    # @example
    #   add( 'password' )
    #   add( 'username', 'password' )
    #
    # @example Add regular expressions
    #   add( /password/ )
    #
    # @return [self] The silencer instance
    def add( *patterns )
      patterns.each { |pattern| add!(pattern) }

      self
    end

    # Clears out all the messages that would match any defined pattern
    #
    # @example
    #   call(['username', 'password'])
    #   #=> ['username]
    #
    # @return [Array] The remaining messages
    def call( *messages )
      return messages if @patterns.empty?

      messages.reject { |m| matches?(m) }
    end

    # Get a pretty string
    def inspect
      "#<#{self.class.name} patterns: #{@patterns.inspect}>"
    end

    # @private
    def patterns
      @patterns
    end


    private

    def add!( pattern )
      @patterns = @patterns | fetch(pattern)
    end

    def fetch( pattern )
      case pattern
      when Symbol then Presets[pattern] or raise PresetNotFound.new(pattern)
      else [pattern]
      end
    end

    # Check if the provided message matches any of the defined patterns.
    #
    # @example
    #   matches?('password')
    #   #=> true
    #
    # @return [Boolean] true or false
    def matches?( message )
      @patterns.any? { |pattern| message.respond_to?(:match) && message.match(pattern) }
    end

  end
end

