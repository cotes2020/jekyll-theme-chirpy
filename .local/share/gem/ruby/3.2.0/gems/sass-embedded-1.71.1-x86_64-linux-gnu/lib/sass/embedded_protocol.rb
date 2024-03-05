# frozen_string_literal: true

module Sass
  # @see https://github.com/sass/sass/blob/HEAD/spec/embedded-protocol.md
  module EmbeddedProtocol
    require_relative '../../ext/sass/embedded_sass_pb'
  end

  private_constant :EmbeddedProtocol
end
