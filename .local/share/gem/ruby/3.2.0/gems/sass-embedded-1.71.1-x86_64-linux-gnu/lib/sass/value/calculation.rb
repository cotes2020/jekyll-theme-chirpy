# frozen_string_literal: true

module Sass
  module Value
    # Sass's calculation type.
    #
    # @see https://sass-lang.com/documentation/js-api/classes/sasscalculation/
    class Calculation
      include Value
      include CalculationValue

      class << self
        private :new

        # @param argument [CalculationValue]
        # @return [Calculation]
        def calc(argument)
          new('calc', [argument])
        end

        # @param arguments [Array<CalculationValue>]
        # @return [Calculation]
        def min(arguments)
          new('min', arguments)
        end

        # @param arguments [Array<CalculationValue>]
        # @return [Calculation]
        def max(arguments)
          new('max', arguments)
        end

        # @param min [CalculationValue]
        # @param value [CalculationValue]
        # @param max [CalculationValue]
        # @return [Calculation]
        def clamp(min, value = nil, max = nil)
          if (value.nil? && !valid_clamp_arg?(min)) ||
             (max.nil? && [min, value].none? { |x| x && valid_clamp_arg?(x) })
            raise Sass::ScriptError, 'Argument must be an unquoted SassString.'
          end

          new('clamp', [min, value, max].compact)
        end

        private

        def valid_clamp_arg?(value)
          value.is_a?(Sass::Value::String) && !value.quoted?
        end
      end

      private

      def initialize(name, arguments)
        arguments.each do |value|
          assert_calculation_value(value)
        end

        @name = name.freeze
        @arguments = arguments.freeze
      end

      public

      # @return [::String]
      attr_reader :name

      # @return [Array<CalculationValue>]
      attr_reader :arguments

      # @return [Calculation]
      def assert_calculation(_name = nil)
        self
      end

      # @return [::Boolean]
      def ==(other)
        other.is_a?(Sass::Value::Calculation) &&
          other.name == name &&
          other.arguments == arguments
      end

      # @return [Integer]
      def hash
        @hash ||= [name, *arguments].hash
      end
    end
  end
end
