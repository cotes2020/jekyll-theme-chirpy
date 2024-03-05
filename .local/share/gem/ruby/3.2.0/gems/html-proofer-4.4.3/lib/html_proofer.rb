# frozen_string_literal: true

require "zeitwerk"
lib_dir = File.join(File.dirname(__dir__), "lib")
gem_loader = Zeitwerk::Loader.for_gem
gem_loader.inflector.inflect(
  "html_proofer" => "HTMLProofer",
)
gem_loader.ignore(File.join(lib_dir, "html-proofer.rb"))
gem_loader.setup

require "html_proofer/version"

require "parallel"
require "fileutils"

if ENV.fetch("DEBUG", false)
  require "awesome_print"
  require "debug"
end

module HTMLProofer
  class << self
    def check_file(file, options = {})
      raise ArgumentError unless file.is_a?(String)
      raise ArgumentError, "#{file} does not exist" unless File.exist?(file)

      options[:type] = :file
      HTMLProofer::Runner.new(file, options)
    end

    def check_directory(directory, options = {})
      raise ArgumentError unless directory.is_a?(String)
      raise ArgumentError, "#{directory} does not exist" unless Dir.exist?(directory)

      options[:type] = :directory
      HTMLProofer::Runner.new([directory], options)
    end

    def check_directories(directories, options = {})
      raise ArgumentError unless directories.is_a?(Array)

      options[:type] = :directory
      directories.each do |directory|
        raise ArgumentError, "#{directory} does not exist" unless Dir.exist?(directory)
      end
      HTMLProofer::Runner.new(directories, options)
    end

    def check_links(links, options = {})
      raise ArgumentError unless links.is_a?(Array)

      options[:type] = :links
      HTMLProofer::Runner.new(links, options)
    end
  end
end

gem_loader.eager_load
