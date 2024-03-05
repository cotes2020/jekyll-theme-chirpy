# frozen_string_literal: true

require 'open3'

require_relative '../../../ext/sass/cli'

module Sass
  class Compiler
    # The stdio based {Connection} between the {Dispatcher} and the compiler.
    #
    # It runs the `sass --embedded` command.
    class Connection
      def initialize(dispatcher)
        @mutex = Mutex.new
        @stdin, @stdout, @stderr, @wait_thread = begin
          Open3.popen3(*CLI::COMMAND, '--embedded', chdir: __dir__)
        rescue Errno::ENOENT
          require_relative '../elf'

          raise if ELF::INTERPRETER.nil?

          Open3.popen3(ELF::INTERPRETER, *CLI::COMMAND, '--embedded', chdir: __dir__)
        end

        @stdin.binmode

        Thread.new do
          Thread.current.name = "sass-embedded-process-stdout-poller-#{@wait_thread.pid}"

          # # https://dart.dev/tools/dart-devtools
          # if 'dart' == File.basename(CLI::COMMAND.first, '.exe') && CLI::COMMAND.include?('--observe')
          #   Kernel.warn(@stdout.readline, uplevel: 0)
          #   Kernel.warn(@stdout.readline, uplevel: 0)
          # end

          @stdout.binmode

          loop do
            length = Varint.read(@stdout)
            id = Varint.read(@stdout)
            proto = @stdout.read(length - Varint.length(id))
            dispatcher.receive_proto(id, proto)
          end
        rescue IOError, Errno::EBADF, Errno::EPROTO => e
          dispatcher.error(e)
          @mutex.synchronize do
            @stdout.close
          end
        end

        Thread.new do
          Thread.current.name = "sass-embedded-process-stderr-poller-#{@wait_thread.pid}"
          loop do
            Kernel.warn(@stderr.readline, uplevel: 0)
          end
        rescue IOError, Errno::EBADF
          @mutex.synchronize do
            @stderr.close
          end
        end

        @wait_thread.name = "sass-embedded-process-waiter-#{@wait_thread.pid}"
      end

      def close
        @mutex.synchronize do
          @stdin.close
          @wait_thread.join
          @stdout.close
          @stderr.close
        end
      end

      def closed?
        @mutex.synchronize do
          @stdin.closed? && !@wait_thread.alive?
        end
      end

      def write(id, proto)
        buffer = []
        Varint.write(buffer, Varint.length(id) + proto.length)
        Varint.write(buffer, id)
        @mutex.synchronize do
          @stdin.write(buffer.pack('C*'), proto)
        end
      end
    end

    private_constant :Connection
  end
end
