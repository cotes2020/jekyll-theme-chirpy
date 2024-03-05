# frozen_string_literal: true

require_relative 'number/unit'

module Sass
  module Value
    # Sass's number type.
    #
    # @see https://sass-lang.com/documentation/js-api/classes/sassnumber/
    class Number
      include Value
      include CalculationValue

      # @param value [Numeric]
      # @param unit [::String, Hash]
      # @option unit [Array<::String>] :numerator_units
      # @option unit [Array<::String>] :denominator_units
      def initialize(value, unit = nil)
        case unit
        when nil
          numerator_units = []
          denominator_units = []
        when ::String
          numerator_units = [unit]
          denominator_units = []
        when ::Hash
          numerator_units = unit.fetch(:numerator_units, [])
          unless numerator_units.is_a?(Array)
            raise Sass::ScriptError, "invalid numerator_units #{numerator_units.inspect}"
          end

          denominator_units = unit.fetch(:denominator_units, [])
          unless denominator_units.is_a?(Array)
            raise Sass::ScriptError, "invalid denominator_units #{denominator_units.inspect}"
          end
        else
          raise Sass::ScriptError, "invalid unit #{unit.inspect}"
        end

        unless denominator_units.empty? && numerator_units.empty?
          value = value.dup
          numerator_units = numerator_units.dup
          new_denominator_units = []

          denominator_units.each do |denominator_unit|
            index = numerator_units.find_index do |numerator_unit|
              factor = Unit.conversion_factor(denominator_unit, numerator_unit)
              if factor.nil?
                false
              else
                value *= factor
                true
              end
            end
            if index.nil?
              new_denominator_units.push(denominator_unit)
            else
              numerator_units.delete_at(index)
            end
          end

          denominator_units = new_denominator_units
        end

        @value = value.freeze
        @numerator_units = numerator_units.each(&:freeze).freeze
        @denominator_units = denominator_units.each(&:freeze).freeze
      end

      # @return [Numeric]
      attr_reader :value

      # @return [Array<::String>]
      attr_reader :numerator_units, :denominator_units

      # @return [::Boolean]
      def ==(other)
        return false unless other.is_a?(Sass::Value::Number)

        return false if numerator_units.length != other.numerator_units.length ||
                        denominator_units.length != other.denominator_units.length

        return FuzzyMath.equals(value, other.value) if unitless?

        if Unit.canonicalize_units(numerator_units) != Unit.canonicalize_units(other.numerator_units) &&
           Unit.canonicalize_units(denominator_units) != Unit.canonicalize_units(other.denominator_units)
          return false
        end

        FuzzyMath.equals(
          (value *
          Unit.canonical_multiplier(numerator_units) /
          Unit.canonical_multiplier(denominator_units)),
          (other.value *
          Unit.canonical_multiplier(other.numerator_units) /
          Unit.canonical_multiplier(other.denominator_units))
        )
      end

      # @return [Integer]
      def hash
        @hash ||= FuzzyMath.hash(canonical_units_value)
      end

      # @return [::Boolean]
      def unitless?
        numerator_units.empty? && denominator_units.empty?
      end

      # @return [Number]
      # @raise [ScriptError]
      def assert_unitless(name = nil)
        raise Sass::ScriptError.new("Expected #{self} to have no units", name) unless unitless?

        self
      end

      # @return [::Boolean]
      def units?
        !unitless?
      end

      # @param unit [::String]
      # @return [::Boolean]
      def unit?(unit)
        single_unit? && numerator_units.first == unit
      end

      # @param unit [::String]
      # @return [Number]
      # @raise [ScriptError]
      def assert_unit(unit, name = nil)
        raise Sass::ScriptError.new("Expected #{self} to have unit #{unit.inspect}", name) unless unit?(unit)

        self
      end

      # @return [::Boolean]
      def integer?
        FuzzyMath.integer?(value)
      end

      # @return [Integer]
      # @raise [ScriptError]
      def assert_integer(name = nil)
        raise Sass::ScriptError.new("#{self} is not an integer", name) unless integer?

        to_i
      end

      # @return [Integer]
      def to_i
        FuzzyMath.to_i(value)
      end

      # @param min [Numeric]
      # @param max [Numeric]
      # @return [Numeric]
      # @raise [ScriptError]
      def assert_between(min, max, name = nil)
        FuzzyMath.assert_between(value, min, max, name)
      end

      # @param unit [::String]
      # @return [::Boolean]
      def compatible_with_unit?(unit)
        single_unit? && !Unit.conversion_factor(numerator_units.first, unit).nil?
      end

      # @param new_numerator_units [Array<::String>]
      # @param new_denominator_units [Array<::String>]
      # @return [Number]
      def convert(new_numerator_units, new_denominator_units, name = nil)
        Number.new(convert_value(new_numerator_units, new_denominator_units, name), {
                     numerator_units: new_numerator_units,
                     denominator_units: new_denominator_units
                   })
      end

      # @param new_numerator_units [Array<::String>]
      # @param new_denominator_units [Array<::String>]
      # @return [Numeric]
      def convert_value(new_numerator_units, new_denominator_units, name = nil)
        coerce_or_convert_value(new_numerator_units, new_denominator_units,
                                coerce_unitless: false,
                                name:)
      end

      # @param other [Number]
      # @return [Number]
      def convert_to_match(other, name = nil, other_name = nil)
        Number.new(convert_value_to_match(other, name, other_name), {
                     numerator_units: other.numerator_units,
                     denominator_units: other.denominator_units
                   })
      end

      # @param other [Number]
      # @return [Numeric]
      def convert_value_to_match(other, name = nil, other_name = nil)
        coerce_or_convert_value(other.numerator_units, other.denominator_units,
                                coerce_unitless: false,
                                name:,
                                other:,
                                other_name:)
      end

      # @param new_numerator_units [Array<::String>]
      # @param new_denominator_units [Array<::String>]
      # @return [Number]
      def coerce(new_numerator_units, new_denominator_units, name = nil)
        Number.new(coerce_value(new_numerator_units, new_denominator_units, name), {
                     numerator_units: new_numerator_units,
                     denominator_units: new_denominator_units
                   })
      end

      # @param new_numerator_units [Array<::String>]
      # @param new_denominator_units [Array<::String>]
      # @return [Numeric]
      def coerce_value(new_numerator_units, new_denominator_units, name = nil)
        coerce_or_convert_value(new_numerator_units, new_denominator_units,
                                coerce_unitless: true,
                                name:)
      end

      # @param unit [::String]
      # @return [Numeric]
      def coerce_value_to_unit(unit, name = nil)
        coerce_value([unit], [], name)
      end

      # @param other [Number]
      # @return [Number]
      def coerce_to_match(other, name = nil, other_name = nil)
        Number.new(coerce_value_to_match(other, name, other_name), {
                     numerator_units: other.numerator_units,
                     denominator_units: other.denominator_units
                   })
      end

      # @param other [Number]
      # @return [Numeric]
      def coerce_value_to_match(other, name = nil, other_name = nil)
        coerce_or_convert_value(other.numerator_units, other.denominator_units,
                                coerce_unitless: true,
                                name:,
                                other:,
                                other_name:)
      end

      # @return [Number]
      def assert_number(_name = nil)
        self
      end

      private

      def single_unit?
        numerator_units.length == 1 && denominator_units.empty?
      end

      def canonical_units_value
        if unitless?
          value
        elsif single_unit?
          value * Unit.canonical_multiplier_for_unit(numerator_units.first)
        else
          value * Unit.canonical_multiplier(numerator_units) / Unit.canonical_multiplier(denominator_units)
        end
      end

      def coerce_or_convert_value(new_numerator_units, new_denominator_units,
                                  coerce_unitless:,
                                  name: nil,
                                  other: nil,
                                  other_name: nil)
        if other && (other.numerator_units != new_denominator_units && other.denominator_units != new_denominator_units)
          raise Sass::ScriptError, "Expect #{other} to have units #{unit_string(new_numerator_units,
                                                                                new_denominator_units).inspect}"
        end

        return value if numerator_units == new_numerator_units && denominator_units == new_denominator_units

        return value if numerator_units == new_numerator_units && denominator_units == new_denominator_units

        other_unitless = new_numerator_units.empty? && new_denominator_units.empty?

        return value if coerce_unitless && (unitless? || other_unitless)

        compatibility_error = lambda {
          unless other.nil?
            message = +"#{self} and"
            message << " $#{other_name}:" unless other_name.nil?
            message << " #{other} have incompatible units"
            message << " (one has units and the other doesn't)" if unitless? || other_unitless
            return Sass::ScriptError.new(message, name)
          end

          return Sass::ScriptError.new("Expected #{self} to have no units", name) unless other_unitless

          if new_numerator_units.length == 1 && new_denominator_units.empty?
            type = Unit::TYPES_BY_UNIT[new_numerator_units.first]
            return Sass::ScriptError.new(
              "Expected #{self} to have a #{type} unit (#{Unit::UNITS_BY_TYPE[type].join(', ')})", name
            )
          end

          unit_length = new_numerator_units.length + new_denominator_units.length
          units = unit_string(new_numerator_units, new_denominator_units)
          Sass::ScriptError.new("Expected #{self} to have unit#{unit_length > 1 ? 's' : ''} #{units}", name)
        }

        result = value

        old_numerator_units = numerator_units.dup
        new_numerator_units.each do |new_numerator_unit|
          index = old_numerator_units.find_index do |old_numerator_unit|
            factor = Unit.conversion_factor(new_numerator_unit, old_numerator_unit)
            if factor.nil?
              false
            else
              result *= factor
              true
            end
          end
          raise compatibility_error.call if index.nil?

          old_numerator_units.delete_at(index)
        end

        old_denominator_units = denominator_units.dup
        new_denominator_units.each do |new_denominator_unit|
          index = old_denominator_units.find_index do |old_denominator_unit|
            factor = Unit.conversion_factor(new_denominator_unit, old_denominator_unit)
            if factor.nil?
              false
            else
              result /= factor
              true
            end
          end
          raise compatibility_error.call if index.nil?

          old_denominator_units.delete_at(index)
        end

        raise compatibility_error.call unless old_numerator_units.empty? && old_denominator_units.empty?

        result
      end

      def unit_string(numerator_units, denominator_units)
        if numerator_units.empty?
          return 'no units' if denominator_units.empty?

          return denominator_units.length == 1 ? "#{denominator_units.first}^-1" : "(#{denominator_units.join('*')})^-1"
        end

        return numerator_units.join('*') if denominator_units.empty?

        "#{numerator_units.join('*')}/#{denominator_units.join('*')}"
      end
    end
  end
end
