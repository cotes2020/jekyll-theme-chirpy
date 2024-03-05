# frozen_string_literal: true

module Sass
  module Value
    # Sass's {FuzzyMath} module.
    module FuzzyMath
      PRECISION = 10

      EPSILON = 10.pow(-PRECISION - 1)

      INVERSE_EPSILON = 1 / EPSILON

      module_function

      def equals(number1, number2)
        (number1 - number2).abs < EPSILON
      end

      def less_than(number1, number2)
        number1 < number2 && !equals(number1, number2)
      end

      def less_than_or_equals(number1, number2)
        number1 < number2 || equals(number1, number2)
      end

      def greater_than(number1, number2)
        number1 > number2 && !equals(number1, number2)
      end

      def greater_than_or_equals(number1, number2)
        number1 > number2 || equals(number1, number2)
      end

      def integer?(number)
        return false unless number.finite?
        return true if number.integer?

        equals((number - 0.5).abs % 1, 0.5)
      end

      def to_i(number)
        integer?(number) ? number.round : nil
      end

      def round(number)
        if number.positive?
          less_than(number % 1, 0.5) ? number.floor : number.ceil
        else
          less_than_or_equals(number % 1, 0.5) ? number.floor : number.ceil
        end
      end

      def between(number, min, max)
        return min if equals(number, min)
        return max if equals(number, max)
        return number if number > min && number < max

        nil
      end

      def assert_between(number, min, max, name)
        result = between(number, min, max)
        return result unless result.nil?

        raise Sass::ScriptError.new("#{number} must be between #{min} and #{max}.", name)
      end

      def hash(number)
        if number.finite?
          (number * INVERSE_EPSILON).round.hash
        else
          number.hash
        end
      end
    end

    private_constant :FuzzyMath
  end
end
