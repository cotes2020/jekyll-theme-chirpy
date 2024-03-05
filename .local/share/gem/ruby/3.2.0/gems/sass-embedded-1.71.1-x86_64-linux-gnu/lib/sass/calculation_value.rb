# frozen_string_literal: true

module Sass
  # The type of values that can be arguments to a SassCalculation.
  #
  # @see https://sass-lang.com/documentation/js-api/types/calculationvalue/
  module CalculationValue
    private

    def assert_calculation_value(value, name = nil)
      if !value.is_a?(Sass::CalculationValue) || (value.is_a?(Sass::Value::String) && value.quoted?)
        raise Sass::ScriptError.new(
          "#{value} must be one of SassNumber, unquoted SassString, SassCalculation, CalculationOperation", name
        )
      end

      value
    end
  end
end

require_relative 'calculation_value/calculation_operation'
