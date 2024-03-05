# frozen_string_literal: true

module Sass
  module Logger
    # A span of text within a source file.
    #
    # @see https://sass-lang.com/documentation/js-api/interfaces/sourcespan/
    class SourceSpan
      # @return [SourceLocation]
      attr_reader :start, :end

      # @return [String]
      attr_reader :text

      # @return [String, nil]
      attr_reader :url, :context

      # @!visibility private
      def initialize(source_span)
        @start = source_span.start.nil? ? nil : Logger::SourceLocation.new(source_span.start)
        @end = source_span.end.nil? ? nil : Logger::SourceLocation.new(source_span.end)
        @text = source_span.text
        @url = source_span.url == '' ? nil : source_span.url
        @context = source_span.context == '' ? nil : source_span.context
      end
    end
  end
end
