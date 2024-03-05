# frozen_string_literal: true

module Sass
  # An exception thrown because a Sass compilation failed.
  class CompileError < StandardError
    # @return [String, nil]
    attr_reader :sass_stack

    # @return [Logger::SourceSpan, nil]
    attr_reader :span

    # @return [Array<String>]
    attr_reader :loaded_urls

    # @!visibility private
    def initialize(message, full_message, sass_stack, span, loaded_urls)
      super(message)

      @full_message = full_message
      @sass_stack = sass_stack
      @span = span
      @loaded_urls = loaded_urls
    end

    # @return [String]
    def full_message(highlight: nil, order: nil, **)
      return super if @full_message.nil?

      highlight = Exception.respond_to?(:to_tty?) && Exception.to_tty? if highlight.nil?
      if highlight
        @full_message.dup
      else
        @full_message.gsub(/\e\[[0-9;]*m/, '')
      end
    end

    # @return [String]
    def to_css
      content = full_message(highlight: false, order: :top)

      <<~CSS.freeze
        /* #{content.gsub('*/', "*\u2060/").gsub("\r\n", "\n").split("\n").join("\n * ")} */

        body::before {
          position: static;
          display: block;
          padding: 1em;
          margin: 0 0 1em;
          border-width: 0 0 2px;
          border-bottom-style: solid;
          font-family: monospace, monospace;
          white-space: pre;
          content: #{Serializer.serialize_quoted_string(content).gsub(/[^[:ascii:]][\h\t ]?/) do |match|
            replacement = "\\#{match.ord.to_s(16)}"
            replacement << " #{match[1]}" if match.length > 1
            replacement
          end};
        }
      CSS
    end
  end

  # An exception thrown by Sass Script.
  class ScriptError < StandardError
    def initialize(message, name = nil)
      super(name.nil? ? message : "$#{name}: #{message}")
    end
  end
end
