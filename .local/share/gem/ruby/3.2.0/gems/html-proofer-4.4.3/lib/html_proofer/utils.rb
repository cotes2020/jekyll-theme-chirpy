# frozen_string_literal: true

require "nokogiri"

module HTMLProofer
  module Utils
    def pluralize(count, single, plural)
      "#{count} #{count == 1 ? single : plural}"
    end

    def blank?(obj)
      obj.nil? || obj.empty?
    end

    def create_nokogiri(path)
      content = if File.exist?(path) && !File.directory?(path)
        File.read(path)
      else
        path
      end

      Nokogiri::HTML5(content, max_errors: -1)
    end
  end
end
