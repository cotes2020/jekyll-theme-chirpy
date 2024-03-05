# frozen_string_literal: true

require_relative 'host/function_registry'
require_relative 'host/importer_registry'
require_relative 'host/logger_registry'
require_relative 'host/protofier'
require_relative 'host/structifier'

module Sass
  class Compiler
    # The {Host} class.
    #
    # It communicates with {Dispatcher} and handles the host logic.
    class Host
      def initialize(channel)
        @channel = channel
      end

      def compile_request(path:,
                          source:,
                          importer:,
                          load_paths:,
                          syntax:,
                          url:,
                          charset:,
                          source_map:,
                          source_map_include_sources:,
                          style:,
                          functions:,
                          importers:,
                          alert_ascii:,
                          alert_color:,
                          logger:,
                          quiet_deps:,
                          verbose:)
        compile_response = await do
          alert_color = Exception.respond_to?(:to_tty?) && Exception.to_tty? if alert_color.nil?

          @function_registry = FunctionRegistry.new(functions, alert_color:)
          @importer_registry = ImporterRegistry.new(importers, load_paths, alert_color:)
          @logger_registry = LoggerRegistry.new(logger)

          send_message(compile_request: EmbeddedProtocol::InboundMessage::CompileRequest.new(
            string: unless source.nil?
                      EmbeddedProtocol::InboundMessage::CompileRequest::StringInput.new(
                        source: source.to_str,
                        url: url&.to_s,
                        syntax: @importer_registry.syntax_to_proto(syntax),
                        importer: (@importer_registry.register(importer) unless importer.nil?)
                      )
                    end,
            path: (File.absolute_path(path) unless path.nil?),
            style: case style&.to_sym
                   when :expanded
                     EmbeddedProtocol::OutputStyle::EXPANDED
                   when :compressed
                     EmbeddedProtocol::OutputStyle::COMPRESSED
                   else
                     raise ArgumentError, 'style must be one of :expanded, :compressed'
                   end,
            charset:,
            source_map:,
            source_map_include_sources:,
            importers: @importer_registry.importers,
            global_functions: @function_registry.global_functions,
            alert_ascii:,
            alert_color:,
            quiet_deps:,
            silent: logger == Logger.silent,
            verbose:
          ))
        end

        oneof = compile_response.result
        result = compile_response.public_send(oneof)
        case oneof
        when :failure
          raise CompileError.new(
            result.message,
            result.formatted == '' ? nil : result.formatted,
            result.stack_trace == '' ? nil : result.stack_trace,
            result.span.nil? ? nil : Logger::SourceSpan.new(result.span),
            compile_response.loaded_urls.to_a
          )
        when :success
          CompileResult.new(
            result.css,
            result.source_map == '' ? nil : result.source_map,
            compile_response.loaded_urls.to_a
          )
        else
          raise ArgumentError, "Unknown CompileResponse.result #{result}"
        end
      end

      def version_request
        version_response = await0 do
          send_message0(version_request: EmbeddedProtocol::InboundMessage::VersionRequest.new(
            id:
          ))
        end

        info = [
          version_response.implementation_name,
          version_response.implementation_version,
          '(Sass Compiler)'
        ]

        case version_response.implementation_name
        when 'dart-sass'
          info << '[Dart]'
        end

        info
      end

      def compile_response(message)
        @result = message
        @queue.close
      end

      def version_response(message)
        @result = message
        @queue.close
      end

      def error(message)
        case message
        when EmbeddedProtocol::ProtocolError
          @error = Errno::EPROTO.new(message.message)
          @stream.error(@error)
        else
          @error ||= message
        end
        @queue.close
      end

      def log_event(message)
        @logger_registry.log(message)
      rescue StandardError => e
        @stream.error(e)
      end

      def canonicalize_request(message)
        send_message(canonicalize_response: @importer_registry.canonicalize(message))
      end

      def import_request(message)
        send_message(import_response: @importer_registry.import(message))
      end

      def file_import_request(message)
        send_message(file_import_response: @importer_registry.file_import(message))
      end

      def function_call_request(message)
        send_message(function_call_response: @function_registry.function_call(message))
      end

      def receive_proto(proto)
        @queue.push(proto)
      end

      private

      def await0
        listen do
          yield

          @queue.pop
        end
      end

      def await
        listen do
          yield

          while (proto = @queue.pop)
            outbound_message = EmbeddedProtocol::OutboundMessage.decode(proto)
            oneof = outbound_message.message
            message = outbound_message.public_send(oneof)
            public_send(oneof, message)
          end
        end
      end

      def listen
        @queue = Queue.new
        @stream = @channel.stream(self)

        yield

        raise @error if @error

        @result
      ensure
        @stream&.close
        @queue&.close
      end

      def id
        @stream.id
      end

      def send_message0(...)
        inbound_message = EmbeddedProtocol::InboundMessage.new(...)
        @stream.send_proto(0, EmbeddedProtocol::InboundMessage.encode(inbound_message))
      end

      def send_message(...)
        inbound_message = EmbeddedProtocol::InboundMessage.new(...)
        @stream.send_proto(id, EmbeddedProtocol::InboundMessage.encode(inbound_message))
      end
    end

    private_constant :Host
  end
end
