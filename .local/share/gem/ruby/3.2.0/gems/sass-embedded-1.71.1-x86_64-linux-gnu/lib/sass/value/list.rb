# frozen_string_literal: true

module Sass
  module Value
    # Sass's list type.
    #
    # @see https://sass-lang.com/documentation/js-api/classes/sasslist/
    class List
      include Value

      # @param contents [Array<Value>]
      # @param separator [::String]
      # @param bracketed [::Boolean]
      def initialize(contents = [], separator: ',', bracketed: false)
        if separator.nil? && contents.length > 1
          raise Sass::ScriptError, 'A list with more than one element must have an explicit separator'
        end

        @contents = contents.freeze
        @separator = separator.freeze
        @bracketed = bracketed.freeze
      end

      # @return [::String, nil]
      attr_reader :separator

      # @return [::Boolean]
      def bracketed?
        @bracketed
      end

      # @return [::Boolean]
      def ==(other)
        (other.is_a?(Sass::Value::List) &&
         other.to_a == to_a &&
         other.separator == separator &&
         other.bracketed? == bracketed?) ||
          (to_a.empty? && other.is_a?(Sass::Value::Map) && other.to_a.empty?)
      end

      # @param index [Numeric]
      # @return [Value]
      def at(index)
        index = index.floor
        index = to_a.length + index if index.negative?
        return nil if index.negative? || index >= to_a.length

        to_a[index]
      end

      # @return [Integer]
      def hash
        @hash ||= contents.hash
      end

      # @return [Array<Value>]
      def to_a
        @contents
      end

      # @return [Map, nil]
      def to_map
        to_a.empty? ? Sass::Value::Map.new({}) : nil
      end

      # @return [Map]
      # @raise [ScriptError]
      def assert_map(name = nil)
        to_a.empty? ? Sass::Value::Map.new({}) : super.assert_map(name)
      end

      private

      def to_a_length
        to_a.length
      end
    end
  end
end
