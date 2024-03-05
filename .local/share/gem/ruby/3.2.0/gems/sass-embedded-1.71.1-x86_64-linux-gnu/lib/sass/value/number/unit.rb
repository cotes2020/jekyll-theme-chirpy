# frozen_string_literal: true

module Sass
  module Value
    class Number
      # The {Unit} module.
      module Unit
        CONVERSIONS = {
          # Length
          'in' => {
            'in' => Rational(1),
            'cm' => Rational(1, 2.54),
            'pc' => Rational(1, 6),
            'mm' => Rational(1, 25.4),
            'q' => Rational(1, 101.6),
            'pt' => Rational(1, 72),
            'px' => Rational(1, 96)
          },
          'cm' => {
            'in' => Rational(2.54),
            'cm' => Rational(1),
            'pc' => Rational(2.54, 6),
            'mm' => Rational(1, 10),
            'q' => Rational(1, 40),
            'pt' => Rational(2.54, 72),
            'px' => Rational(2.54, 96)
          },
          'pc' => {
            'in' => Rational(6),
            'cm' => Rational(6, 2.54),
            'pc' => Rational(1),
            'mm' => Rational(6, 25.4),
            'q' => Rational(6, 101.6),
            'pt' => Rational(1, 12),
            'px' => Rational(1, 16)
          },
          'mm' => {
            'in' => Rational(25.4),
            'cm' => Rational(10),
            'pc' => Rational(25.4, 6),
            'mm' => Rational(1),
            'q' => Rational(1, 4),
            'pt' => Rational(25.4, 72),
            'px' => Rational(25.4, 96)
          },
          'q' => {
            'in' => Rational(101.6),
            'cm' => Rational(40),
            'pc' => Rational(101.6, 6),
            'mm' => Rational(4),
            'q' => Rational(1),
            'pt' => Rational(101.6, 72),
            'px' => Rational(101.6, 96)
          },
          'pt' => {
            'in' => Rational(72),
            'cm' => Rational(72, 2.54),
            'pc' => Rational(12),
            'mm' => Rational(72, 25.4),
            'q' => Rational(72, 101.6),
            'pt' => Rational(1),
            'px' => Rational(3, 4)
          },
          'px' => {
            'in' => Rational(96),
            'cm' => Rational(96, 2.54),
            'pc' => Rational(16),
            'mm' => Rational(96, 25.4),
            'q' => Rational(96, 101.6),
            'pt' => Rational(4, 3),
            'px' => Rational(1)
          },

          # Rotation
          'deg' => {
            'deg' => Rational(1),
            'grad' => Rational(9, 10),
            'rad' => Rational(180, Math::PI),
            'turn' => Rational(360)
          },
          'grad' => {
            'deg' => Rational(10, 9),
            'grad' => Rational(1),
            'rad' => Rational(200, Math::PI),
            'turn' => Rational(400)
          },
          'rad' => {
            'deg' => Rational(Math::PI, 180),
            'grad' => Rational(Math::PI, 200),
            'rad' => Rational(1),
            'turn' => Rational(Math::PI * 2)
          },
          'turn' => {
            'deg' => Rational(1, 360),
            'grad' => Rational(1, 400),
            'rad' => Rational(1, Math::PI * 2),
            'turn' => Rational(1)
          },

          # Time
          's' => {
            's' => Rational(1),
            'ms' => Rational(1, 1000)
          },
          'ms' => {
            's' => Rational(1000),
            'ms' => Rational(1)
          },

          # Frequency
          'Hz' => {
            'Hz' => Rational(1),
            'kHz' => Rational(1000)
          },
          'kHz' => {
            'Hz' => Rational(1, 1000),
            'kHz' => Rational(1)
          },

          # Pixel density
          'dpi' => {
            'dpi' => Rational(1),
            'dpcm' => Rational(2.54),
            'dppx' => Rational(96)
          },
          'dpcm' => {
            'dpi' => Rational(1, 2.54),
            'dpcm' => Rational(1),
            'dppx' => Rational(96, 2.54)
          },
          'dppx' => {
            'dpi' => Rational(1, 96),
            'dpcm' => Rational(2.54, 96),
            'dppx' => Rational(1)
          }
        }.freeze

        UNITS_BY_TYPE = {
          time: %w[s ms],
          frequency: %w[Hz kHz],
          'pixel density': %w[dpi dpcm dppx]
        }.freeze

        TYPES_BY_UNIT = UNITS_BY_TYPE.invert
                                     .to_a
                                     .flat_map { |pair| pair[0].map { |key| [key, pair[1]] } }
                                     .to_h

        module_function

        def conversion_factor(unit1, unit2)
          return 1 if unit1 == unit2

          CONVERSIONS.dig(unit1, unit2)
        end

        def canonicalize_units(units)
          return units if units.empty?

          if units.length == 1
            type = TYPES_BY_UNIT[units.first]
            return type.nil? ? units : [UNITS_BY_TYPE[type].first]
          end

          units.map do |unit|
            type = TYPES_BY_UNIT[unit]
            type.nil? ? units : [UNITS_BY_TYPE[type].first]
          end.sort
        end

        def canonical_multiplier(units)
          units.reduce(1) do |multiplier, unit|
            multiplier * canonical_multiplier_for_unit(unit)
          end
        end

        def canonical_multiplier_for_unit(unit)
          inner_map = CONVERSIONS[unit]
          inner_map.nil? ? 1 : 1 / inner_map.values.first
        end
      end

      private_constant :Unit
    end
  end
end
