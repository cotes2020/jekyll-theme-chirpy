module Yell #:nodoc:
  module Adapters #:nodoc:
    class Io < Yell::Adapters::Base
      include Yell::Helpers::Formatter

      # The possible unix log colors
      TTYColors = {
        0   => "\033[1;32m",  # green
        1   => "\033[0m",     # normal
        2   => "\033[1;33m",  # yellow
        3   => "\033[1;31m",  # red
        4   => "\033[1;35m",  # magenta
        5   => "\033[1;36m",  # cyan
        -1  => "\033[0m"      # normal
      }

      # Sets the “sync mode” to true or false.
      #
      # When true (default), every log event is immediately written to the file. 
      # When false, the log event is buffered internally.
      attr_accessor :sync

      # Sets colored output on or off (default off)
      #
      # @example Enable colors
      #   colors = true
      #
      # @example Disable colors
      #   colors = false
      attr_accessor :colors

      # Shortcut to enable colors.
      #
      # @example
      #   colorize!
      def colorize!; @colors = true; end


      private

      # @overload setup!( options )
      def setup!( options )
        @stream = nil

        self.colors = Yell.__fetch__(options, :colors, default: false)
        self.formatter = Yell.__fetch__(options, :format, :formatter)
        self.sync = Yell.__fetch__(options, :sync, default: true)

        super
      end

      # @overload write!( event )
      def write!( event )
        message = formatter.call(event)

        # colorize if applicable
        if colors and color = TTYColors[event.level]
          message = color + message + TTYColors[-1]
        end

        stream.syswrite(message)

        super
      end

      # @overload open!
      def open!
        @stream.sync = self.sync if @stream.respond_to?(:sync)
        @stream.flush if @stream.respond_to?(:flush)

        super
      end

      # @overload close!
      def close!
        @stream.close if @stream.respond_to?(:close)
        @stream = nil

        super
      end

      # The IO stream
      #
      # Adapter classes should provide their own implementation 
      # of this method.
      def stream
        synchronize { open! if @stream.nil?; @stream }
      end

      # @overload inspectables
      def inspectables
        super.concat [:formatter, :colors, :sync]
      end
    end
  end
end

