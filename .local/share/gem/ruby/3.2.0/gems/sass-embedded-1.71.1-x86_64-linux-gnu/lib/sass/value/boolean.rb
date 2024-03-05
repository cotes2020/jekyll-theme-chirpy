# frozen_string_literal: true

module Sass
  module Value
    # Sass's boolean type.
    #
    # @see https://sass-lang.com/documentation/js-api/classes/sassboolean/
    class Boolean
      include Value

      # @param value [::Boolean]
      def initialize(value)
        @value = value
      end

      # @return [::Boolean]
      attr_reader :value

      # @return [Boolean]
      def !
        value ? Boolean::FALSE : Boolean::TRUE
      end

      # @return [::Boolean]
      def ==(other)
        other.is_a?(Sass::Value::Boolean) && other.value == value
      end

      # @return [Integer]
      def hash
        @hash ||= value.hash
      end

      alias to_bool value

      # @return [Boolean]
      def assert_boolean(_name = nil)
        self
      end

      # Sass's true value.
      TRUE = Boolean.new(true)

      # Sass's false value.
      FALSE = Boolean.new(false)

      def self.new(value)
        value ? Boolean::TRUE : Boolean::FALSE
      end
    end
  end
end
