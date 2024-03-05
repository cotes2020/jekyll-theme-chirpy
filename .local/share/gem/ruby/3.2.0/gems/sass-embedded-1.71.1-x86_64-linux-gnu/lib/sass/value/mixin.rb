# frozen_string_literal: true

module Sass
  module Value
    # Sass's mixin type.
    #
    # @see https://sass-lang.com/documentation/js-api/classes/sassmixin/
    class Mixin
      include Value

      class << self
        private :new
      end

      # @return [Integer]
      attr_reader :id

      protected :id

      # @return [::Boolean]
      def ==(other)
        other.is_a?(Sass::Value::Mixin) && other.id == id
      end

      # @return [Integer]
      def hash
        @hash ||= id.hash
      end

      # @return [Mixin]
      def assert_mixin(_name = nil)
        self
      end
    end
  end
end
