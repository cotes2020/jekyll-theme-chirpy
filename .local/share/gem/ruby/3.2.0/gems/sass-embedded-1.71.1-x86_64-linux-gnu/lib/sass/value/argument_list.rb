# frozen_string_literal: true

module Sass
  module Value
    # Sass's argument list type.
    #
    # An argument list comes from a rest argument. It's distinct from a normal {List} in that it may contain a keyword
    # map as well as the positional arguments.
    #
    # @see https://sass-lang.com/documentation/js-api/classes/sassargumentlist/
    class ArgumentList < List
      # @param contents [Array<Value>]
      # @param keywords [Hash<Symbol, Value>]
      # @param separator [::String]
      def initialize(contents = [], keywords = {}, separator = ',')
        super(contents, separator:)

        @id = 0
        @keywords_accessed = false
        @keywords = keywords.freeze
      end

      # @return [Hash<Symbol, Value>]
      def keywords
        @keywords_accessed = true
        @keywords
      end

      private

      def initialize_copy(orig)
        super
        @id = 0
      end
    end
  end
end
