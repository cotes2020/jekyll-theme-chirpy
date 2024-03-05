# frozen_string_literal: true

module Sass
  module Value
    # Sass's function type.
    #
    # @see https://sass-lang.com/documentation/js-api/classes/sassfunction/
    class Function
      include Value

      # @param signature [::String]
      # @param callback [Proc]
      def initialize(signature, &callback)
        raise Sass::ScriptError, 'no block given' unless signature.nil? || callback

        @signature = signature.freeze
        @callback = callback.freeze
      end

      # @return [Integer, nil]
      attr_reader :id

      protected :id

      # @return [::String, nil]
      attr_reader :signature

      # @return [Proc, nil]
      attr_reader :callback

      # @return [::Boolean]
      def ==(other)
        if id.nil?
          other.equal?(self)
        else
          other.is_a?(Sass::Value::Function) && other.id == id
        end
      end

      # @return [Integer]
      def hash
        @hash ||= id.nil? ? signature.hash : id.hash
      end

      # @return [Function]
      def assert_function(_name = nil)
        self
      end
    end
  end
end
