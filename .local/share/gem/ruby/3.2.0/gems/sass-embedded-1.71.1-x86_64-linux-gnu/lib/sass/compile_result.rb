# frozen_string_literal: true

module Sass
  # The result of compiling Sass to CSS. Returned by {Sass.compile} and {Sass.compile_string}.
  #
  # @see https://sass-lang.com/documentation/js-api/interfaces/compileresult/
  class CompileResult
    # @return [String]
    attr_reader :css

    # @return [String, nil]
    attr_reader :source_map

    # @return [Array<String>]
    attr_reader :loaded_urls

    # @!visibility private
    def initialize(css, source_map, loaded_urls)
      @css = css
      @source_map = source_map
      @loaded_urls = loaded_urls
    end
  end
end
