# frozen_string_literal: true

module Sass
  class Compiler
    # The {Dispatcher} class.
    #
    # It dispatches messages between multiple instances of {Host} and a single {Connection} to the compiler.
    class Dispatcher
      def initialize
        @id = 1
        @observers = {}.compare_by_identity
        @mutex = Mutex.new
        @connection = Connection.new(self)
        ForkTracker.add(self)
      end

      def subscribe(observer)
        @mutex.synchronize do
          raise Errno::EBUSY if _closed?

          id = @id
          @id = id.next
          @observers[id] = observer
          id
        end
      end

      def unsubscribe(id)
        @mutex.synchronize do
          @observers.delete(id)

          return unless @observers.empty?

          if _closed?
            Thread.new do
              close
            end
          else
            _idle
          end
        end
      end

      def close
        @mutex.synchronize do
          _close
        end
        @connection.close
        ForkTracker.delete(self)
      end

      def closed?
        @connection.closed?
      end

      def error(error)
        observers = @mutex.synchronize do
          _close
          @observers.values
        end

        if observers.empty?
          close
        else
          observers.each do |observer|
            observer.error(error)
          end
        end
      end

      def receive_proto(id, proto)
        case id
        when 1...0xffffffff
          @mutex.synchronize { @observers[id] }&.receive_proto(proto)
        when 0
          outbound_message = EmbeddedProtocol::OutboundMessage.decode(proto)
          oneof = outbound_message.message
          message = outbound_message.public_send(oneof)
          @mutex.synchronize { @observers[message.id] }&.public_send(oneof, message)
        when 0xffffffff
          outbound_message = EmbeddedProtocol::OutboundMessage.decode(proto)
          oneof = outbound_message.message
          message = outbound_message.public_send(oneof)
          raise Errno::EPROTO, message.message
        else
          raise Errno::EPROTO
        end
      end

      def send_proto(...)
        @connection.write(...)
      end

      private

      def _close
        @id = 0xffffffff
      end

      def _closed?
        @id == 0xffffffff
      end

      def _idle
        @id = 1
      end

      def _idle?
        @id == 1
      end
    end

    private_constant :Dispatcher
  end
end
