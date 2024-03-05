# frozen_string_literal: true

module Sass
  module Value
    # Sass's null type.
    #
    # @see https://sass-lang.com/documentation/js-api/variables/sassnull/
    class Null
      include Value

      def initialize
        @value = nil
      end

      # @return [nil]
      attr_reader :value

      # @return [Boolean]
      def !
        Boolean::TRUE
      end

      # @return [::Boolean]
      def ==(other)
        other.is_a?(Sass::Value::Null)
      end

      # @return [Integer]
      def hash
        @hash ||= value.hash
      end

      # @return [::Boolean]
      def to_bool
        false
      end

      alias to_nil value

      # Sass's null value.
      NULL = Null.new

      def self.new
        NULL
      end
    end
  end
end
