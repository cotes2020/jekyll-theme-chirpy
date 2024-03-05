module Yell #:nodoc:
  module Helpers #:nodoc:
    module Tracer #:nodoc:

      # Set whether the logger should allow tracing or not. The trace option
      # will tell the logger when to provider caller information.
      #
      # @example No tracing at all
      #   trace = false
      #
      # @example Trace every time
      #   race = true
      #
      # @example Trace from the error level onwards
      #   trace = :error
      #   trace = 'gte.error'
      #
      # @return [Yell::Level] a level representation of the tracer
      def trace=( severity )
        @__trace__ = case severity
        when Yell::Level then severity
        when false then Yell::Level.new("gt.#{Yell::Severities.last}")
        else Yell::Level.new(severity)
        end
      end

      def trace
        @__trace__
      end


      private

      def reset!
        @__trace__ = Yell::Level.new('gte.error')

        super
      end

      def inspectables
        [:trace] | super
      end

    end
  end
end

