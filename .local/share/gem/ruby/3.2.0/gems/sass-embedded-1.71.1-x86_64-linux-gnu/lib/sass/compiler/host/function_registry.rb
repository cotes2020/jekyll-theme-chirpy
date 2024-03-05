# frozen_string_literal: true

module Sass
  class Compiler
    class Host
      # The {FunctionRegistry} class.
      #
      # It stores sass custom functions and handles function calls.
      class FunctionRegistry
        attr_reader :global_functions

        def initialize(functions, alert_color:)
          functions = functions.transform_keys(&:to_s)

          @global_functions = functions.keys
          @functions_by_name = functions.transform_keys do |signature|
            index = signature.index('(')
            if index
              signature.slice(0, index)
            else
              signature
            end
          end

          @id = 0
          @functions_by_id = {}.compare_by_identity
          @ids_by_function = {}.compare_by_identity

          @highlight = alert_color
        end

        def register(function)
          @ids_by_function.fetch(function) do |fn|
            id = @id
            @id = id.next

            @functions_by_id[id] = fn
            @ids_by_function[fn] = id
          end
        end

        def function_call(function_call_request)
          oneof = function_call_request.identifier
          identifier = function_call_request.public_send(oneof)
          function = case oneof
                     when :name
                       @functions_by_name[identifier]
                     when :function_id
                       @functions_by_id[identifier]
                     else
                       raise ArgumentError, "Unknown FunctionCallRequest.identifier #{identifier}"
                     end

          arguments = function_call_request.arguments.map do |argument|
            protofier.from_proto(argument)
          end

          success = protofier.to_proto(function.call(arguments))
          accessed_argument_lists = arguments.filter_map do |argument|
            if argument.is_a?(Sass::Value::ArgumentList) && argument.instance_eval { @keywords_accessed }
              argument.instance_eval { @id }
            end
          end

          EmbeddedProtocol::InboundMessage::FunctionCallResponse.new(
            id: function_call_request.id,
            success:,
            accessed_argument_lists:
          )
        rescue StandardError => e
          EmbeddedProtocol::InboundMessage::FunctionCallResponse.new(
            id: function_call_request.id,
            error: e.full_message(highlight: @highlight, order: :top)
          )
        end

        private

        def protofier
          @protofier ||= Protofier.new(self)
        end
      end

      private_constant :FunctionRegistry
    end
  end
end
