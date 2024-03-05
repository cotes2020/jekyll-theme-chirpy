# frozen_string_literal: true

module Sass
  module Value
    # Sass's map type.
    #
    # @see https://sass-lang.com/documentation/js-api/classes/sassmap/
    class Map
      include Value

      # @param contents [Hash<Value, Value>]
      def initialize(contents = {})
        @contents = contents.freeze
      end

      # @return [Hash<Value, Value>]
      attr_reader :contents

      # @return [::String, nil]
      def separator
        contents.empty? ? nil : ','
      end

      # @return [::Boolean]
      def ==(other)
        (other.is_a?(Sass::Value::Map) && other.contents == contents) ||
          (contents.empty? && other.is_a?(Sass::Value::List) && other.to_a.empty?)
      end

      # @param index [Numeric, Value]
      # @return [List<(Value, Value)>, Value]
      def at(index)
        if index.is_a?(Numeric)
          index = index.floor
          index = to_a_length + index if index.negative?
          return nil if index.negative? || index >= to_a_length

          Sass::Value::List.new(contents.to_a[index], separator: ' ')
        else
          contents[index]
        end
      end

      # @return [Integer]
      def hash
        @hash ||= contents.hash
      end

      # @return [Array<List<(Value, Value)>>]
      def to_a
        contents.map { |key, value| Sass::Value::List.new([key, value], separator: ' ') }
      end

      # @return [Map]
      def to_map
        self
      end

      # @return [Map]
      def assert_map(_name = nil)
        self
      end

      private

      def to_a_length
        contents.length
      end
    end
  end
end
