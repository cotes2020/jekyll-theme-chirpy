# frozen_string_literal: true

module Sass
  class Compiler
    # The {Channel} class.
    #
    # It manages the lifecycle of {Dispatcher}.
    class Channel
      def initialize(dispatcher_class)
        @dispatcher_class = dispatcher_class
        @dispatcher = @dispatcher_class.new
        @mutex = Mutex.new
      end

      def close
        @mutex.synchronize do
          unless @dispatcher.nil?
            @dispatcher.close
            @dispatcher = nil
          end
        end
      end

      def closed?
        @mutex.synchronize do
          @dispatcher.nil?
        end
      end

      def stream(host)
        @mutex.synchronize do
          raise IOError, 'closed compiler' if @dispatcher.nil?

          Stream.new(@dispatcher, host)
        rescue Errno::EBUSY
          @dispatcher = @dispatcher_class.new
          Stream.new(@dispatcher, host)
        end
      end

      # The {Stream} between {Dispatcher} and {Host}.
      class Stream
        attr_reader :id

        def initialize(dispatcher, host)
          @dispatcher = dispatcher
          @id = @dispatcher.subscribe(host)
        end

        def close
          @dispatcher.unsubscribe(@id)
        end

        def error(...)
          @dispatcher.error(...)
        end

        def send_proto(...)
          @dispatcher.send_proto(...)
        end
      end

      private_constant :Stream
    end

    private_constant :Channel
  end
end
