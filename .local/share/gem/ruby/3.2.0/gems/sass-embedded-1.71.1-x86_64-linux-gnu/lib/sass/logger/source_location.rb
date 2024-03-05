# frozen_string_literal: true

module Sass
  module Logger
    # A specific location within a source file.
    #
    # This is always associated with a {SourceSpan} which indicates which file it refers to.
    #
    # @see https://sass-lang.com/documentation/js-api/interfaces/sourcelocation/
    class SourceLocation
      # @return [Integer]
      attr_reader :offset, :line, :column

      # @!visibility private
      def initialize(source_location)
        @offset = source_location.offset
        @line = source_location.line
        @column = source_location.column
      end
    end
  end
end
