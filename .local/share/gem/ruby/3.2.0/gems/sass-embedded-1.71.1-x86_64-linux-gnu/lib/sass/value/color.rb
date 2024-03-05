# frozen_string_literal: true

module Sass
  module Value
    # Sass's color type.
    #
    # No matter what representation was originally used to create this color, all of its channels are accessible.
    #
    # @see https://sass-lang.com/documentation/js-api/classes/sasscolor/
    class Color
      include Value

      # @param red [Numeric]
      # @param green [Numeric]
      # @param blue [Numeric]
      # @param hue [Numeric]
      # @param saturation [Numeric]
      # @param lightness [Numeric]
      # @param whiteness [Numeric]
      # @param blackness [Numeric]
      # @param alpha [Numeric]
      def initialize(red: nil,
                     green: nil,
                     blue: nil,
                     hue: nil,
                     saturation: nil,
                     lightness: nil,
                     whiteness: nil,
                     blackness: nil,
                     alpha: 1)
        @alpha = alpha.nil? ? 1 : FuzzyMath.assert_between(alpha, 0, 1, 'alpha')
        if red && green && blue
          @red = FuzzyMath.assert_between(FuzzyMath.round(red), 0, 255, 'red')
          @green = FuzzyMath.assert_between(FuzzyMath.round(green), 0, 255, 'green')
          @blue = FuzzyMath.assert_between(FuzzyMath.round(blue), 0, 255, 'blue')
        elsif hue && saturation && lightness
          @hue = hue % 360
          @saturation = FuzzyMath.assert_between(saturation, 0, 100, 'saturation')
          @lightness = FuzzyMath.assert_between(lightness, 0, 100, 'lightness')
        elsif hue && whiteness && blackness
          @hue = hue % 360
          @whiteness = FuzzyMath.assert_between(whiteness, 0, 100, 'whiteness')
          @blackness = FuzzyMath.assert_between(blackness, 0, 100, 'blackness')
          hwb_to_rgb
          @whiteness = @blackness = nil
        else
          raise Sass::ScriptError, 'Invalid Color'
        end
      end

      # @return [Integer]
      def red
        hsl_to_rgb unless defined?(@red)

        @red
      end

      # @return [Integer]
      def green
        hsl_to_rgb unless defined?(@green)

        @green
      end

      # @return [Integer]
      def blue
        hsl_to_rgb unless defined?(@blue)

        @blue
      end

      # @return [Numeric]
      def hue
        rgb_to_hsl unless defined?(@hue)

        @hue
      end

      # @return [Numeric]
      def saturation
        rgb_to_hsl unless defined?(@saturation)

        @saturation
      end

      # @return [Numeric]
      def lightness
        rgb_to_hsl unless defined?(@lightness)

        @lightness
      end

      # @return [Numeric]
      def whiteness
        @whiteness ||= Rational([red, green, blue].min, 255) * 100
      end

      # @return [Numeric]
      def blackness
        @blackness ||= 100 - (Rational([red, green, blue].max, 255) * 100)
      end

      # @return [Numeric]
      attr_reader :alpha

      # @param red [Numeric]
      # @param green [Numeric]
      # @param blue [Numeric]
      # @param hue [Numeric]
      # @param saturation [Numeric]
      # @param lightness [Numeric]
      # @param whiteness [Numeric]
      # @param blackness [Numeric]
      # @param alpha [Numeric]
      # @return [Color]
      def change(red: nil,
                 green: nil,
                 blue: nil,
                 hue: nil,
                 saturation: nil,
                 lightness: nil,
                 whiteness: nil,
                 blackness: nil,
                 alpha: nil)
        if whiteness || blackness
          Sass::Value::Color.new(hue: hue || self.hue,
                                 whiteness: whiteness || self.whiteness,
                                 blackness: blackness || self.blackness,
                                 alpha: alpha || self.alpha)
        elsif hue || saturation || lightness
          Sass::Value::Color.new(hue: hue || self.hue,
                                 saturation: saturation || self.saturation,
                                 lightness: lightness || self.lightness,
                                 alpha: alpha || self.alpha)
        elsif red || green || blue
          Sass::Value::Color.new(red: red ? FuzzyMath.round(red) : self.red,
                                 green: green ? FuzzyMath.round(green) : self.green,
                                 blue: blue ? FuzzyMath.round(blue) : self.blue,
                                 alpha: alpha || self.alpha)
        else
          dup.instance_eval do
            @alpha = FuzzyMath.assert_between(alpha, 0, 1, 'alpha')
            self
          end
        end
      end

      # @return [::Boolean]
      def ==(other)
        other.is_a?(Sass::Value::Color) &&
          other.red == red &&
          other.green == green &&
          other.blue == blue &&
          other.alpha == alpha
      end

      # @return [Integer]
      def hash
        @hash ||= [red, green, blue, alpha].hash
      end

      # @return [Color]
      def assert_color(_name = nil)
        self
      end

      private

      def rgb_to_hsl
        scaled_red = Rational(red, 255)
        scaled_green = Rational(green, 255)
        scaled_blue = Rational(blue, 255)

        max = [scaled_red, scaled_green, scaled_blue].max
        min = [scaled_red, scaled_green, scaled_blue].min
        delta = max - min

        if max == min
          @hue = 0
        elsif max == scaled_red
          @hue = ((scaled_green - scaled_blue) * 60 / delta) % 360
        elsif max == scaled_green
          @hue = (((scaled_blue - scaled_red) * 60 / delta) + 120) % 360
        elsif max == scaled_blue
          @hue = (((scaled_red - scaled_green) * 60 / delta) + 240) % 360
        end

        lightness = @lightness = (max + min) * 50

        @saturation = if max == min
                        0
                      elsif lightness < 50
                        delta * 100 / (max + min)
                      else
                        delta * 100 / (2 - max - min)
                      end
      end

      def hsl_to_rgb
        scaled_hue = Rational(hue, 360)
        scaled_saturation = Rational(saturation, 100)
        scaled_lightness = Rational(lightness, 100)

        tmp2 = if scaled_lightness <= 0.5
                 scaled_lightness * (scaled_saturation + 1)
               else
                 scaled_lightness + scaled_saturation - (scaled_lightness * scaled_saturation)
               end
        tmp1 = (scaled_lightness * 2) - tmp2
        @red = FuzzyMath.round(hsl_hue_to_rgb(tmp1, tmp2, scaled_hue + Rational(1, 3)) * 255)
        @green = FuzzyMath.round(hsl_hue_to_rgb(tmp1, tmp2, scaled_hue) * 255)
        @blue = FuzzyMath.round(hsl_hue_to_rgb(tmp1, tmp2, scaled_hue - Rational(1, 3)) * 255)
      end

      def hsl_hue_to_rgb(tmp1, tmp2, hue)
        hue += 1 if hue.negative?
        hue -= 1 if hue > 1

        if hue < Rational(1, 6)
          tmp1 + ((tmp2 - tmp1) * hue * 6)
        elsif hue < Rational(1, 2)
          tmp2
        elsif hue < Rational(2, 3)
          tmp1 + ((tmp2 - tmp1) * (Rational(2, 3) - hue) * 6)
        else
          tmp1
        end
      end

      def hwb_to_rgb
        scaled_hue = Rational(hue, 360)
        scaled_whiteness = Rational(whiteness, 100)
        scaled_blackness = Rational(blackness, 100)

        sum = scaled_whiteness + scaled_blackness
        if sum > 1
          scaled_whiteness /= sum
          scaled_blackness /= sum
        end

        factor = 1 - scaled_whiteness - scaled_blackness
        @red = hwb_hue_to_rgb(factor, scaled_whiteness, scaled_hue + Rational(1, 3))
        @green = hwb_hue_to_rgb(factor, scaled_whiteness, scaled_hue)
        @blue = hwb_hue_to_rgb(factor, scaled_whiteness, scaled_hue - Rational(1, 3))
      end

      def hwb_hue_to_rgb(factor, scaled_whiteness, scaled_hue)
        channel = (hsl_hue_to_rgb(0, 1, scaled_hue) * factor) + scaled_whiteness
        FuzzyMath.round(channel * 255)
      end
    end
  end
end
