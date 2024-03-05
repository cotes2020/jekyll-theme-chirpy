# frozen_string_literal: true

module HTMLProofer
  # Represents an element currently being processed
  class Attribute
    include HTMLProofer::Utils

    attr_reader :raw_attribute

    def initialize(runner, raw_attribute, **_)
      @runner = runner
      @raw_attribute = raw_attribute
    end
  end
end
