# frozen_string_literal: true

# stdlib
require "json"

# 3rd party
require "addressable/uri"
require "sass-embedded"

# internal
require_relative "../source_map_page"

module Jekyll
  module Converters
    class Scss < Converter
      EXTENSION_PATTERN = %r!^\.scss$!i.freeze

      SyntaxError = Class.new(ArgumentError)

      safe true
      priority :low

      # This hook is triggered just before the method {#convert(content)} is executed, it
      # associates the Scss (and Sass) converters with their respective sass_page objects.
      Jekyll::Hooks.register :pages, :pre_render do |page|
        next unless page.is_a?(Jekyll::Page)

        page.converters.each do |converter|
          converter.associate_page(page) if converter.is_a?(Jekyll::Converters::Scss)
        end
      end

      # This hook is triggered just after the method {#convert(content)} has been executed, it
      # dissociates the Scss (and Sass) converters with their respective sass_page objects.
      Jekyll::Hooks.register :pages, :post_render do |page|
        next unless page.is_a?(Jekyll::Page)

        page.converters.each do |converter|
          converter.dissociate_page(page) if converter.is_a?(Jekyll::Converters::Scss)
        end
      end

      ALLOWED_STYLES = %w(expanded compressed).freeze

      # Associate this Converter with the "page" object that manages input and output files for
      # this converter.
      #
      # Note: changing the associated sass_page during the live time of this Converter instance
      # may result in inconsistent results.
      #
      # @param [Jekyll:Page] page The sass_page for which this object acts as converter.
      def associate_page(page)
        if @sass_page
          Jekyll.logger.debug "Sass Converter:",
                              "sass_page re-assigned: #{@sass_page.name} to #{page.name}"
          dissociate_page(page)
          return
        end
        @sass_page = page
      end

      # Dissociate this Converter with the "page" object.
      #
      # @param [Jekyll:Page] page The sass_page for which this object has acted as a converter.
      def dissociate_page(page)
        unless page.equal?(@sass_page)
          Jekyll.logger.debug "Sass Converter:",
                              "dissociating a page that was never associated #{page.name}"
        end

        @source_map_page = nil
        @sass_page = nil
        @site = nil
      end

      def matches(ext)
        ext =~ self.class::EXTENSION_PATTERN
      end

      def output_ext(_ext)
        ".css"
      end

      def safe?
        !!@config["safe"]
      end

      def jekyll_sass_configuration
        @jekyll_sass_configuration ||= begin
          options = @config["sass"] || {}
          unless options["style"].nil?
            options["style"] = options["style"].to_s.delete_prefix(":").to_sym
          end
          options
        end
      end

      def syntax
        :scss
      end

      def sass_dir
        return "_sass" if jekyll_sass_configuration["sass_dir"].to_s.empty?

        jekyll_sass_configuration["sass_dir"]
      end

      def sass_style
        style = jekyll_sass_configuration["style"]
        ALLOWED_STYLES.include?(style.to_s) ? style.to_sym : :expanded
      end

      def user_sass_load_paths
        Array(jekyll_sass_configuration["load_paths"])
      end

      def sass_dir_relative_to_site_source
        @sass_dir_relative_to_site_source ||=
          Jekyll.sanitized_path(site_source, sass_dir).sub(site.source + "/", "")
      end

      # rubocop:disable Metrics/AbcSize
      def sass_load_paths
        paths = user_sass_load_paths + [sass_dir_relative_to_site_source]

        # Sanitize paths to prevent any attack vectors (.e.g. `/**/*`)
        paths.map! { |path| Jekyll.sanitized_path(site_source, path) } if safe?

        # Expand file globs (e.g. `node_modules/*/node_modules` )
        Dir.chdir(site_source) do
          paths = paths.flat_map { |path| Dir.glob(path) }

          paths.map! do |path|
            # Sanitize again in case globbing was able to do something crazy.
            safe? ? Jekyll.sanitized_path(site_source, path) : File.expand_path(path)
          end
        end

        paths.uniq!
        paths << site.theme.sass_path if site.theme&.sass_path
        paths.select { |path| File.directory?(path) }
      end
      # rubocop:enable Metrics/AbcSize

      def sass_configs
        {
          :load_paths                 => sass_load_paths,
          :charset                    => !associate_page_failed?,
          :source_map                 => sourcemap_required?,
          :source_map_include_sources => true,
          :style                      => sass_style,
          :syntax                     => syntax,
          :url                        => sass_file_url,
          :quiet_deps                 => quiet_deps_option,
          :verbose                    => verbose_option,
        }
      end

      def convert(content)
        output = ::Sass.compile_string(content, **sass_configs)
        result = output.css

        if sourcemap_required?
          source_map = process_source_map(output.source_map)
          generate_source_map_page(source_map)

          if (sm_url = source_mapping_url)
            result += "#{sass_style == :compressed ? "" : "\n\n"}/*# sourceMappingURL=#{sm_url} */"
          end
        end

        result
      rescue ::Sass::CompileError => e
        Jekyll.logger.error e.full_message
        raise SyntaxError, e.message
      end

      private

      # The Page instance for which this object acts as a converter.
      attr_reader :sass_page

      def associate_page_failed?
        !sass_page
      end

      # The URL of the input scss (or sass) file. This information will be used for error reporting.
      def sass_file_url
        return if associate_page_failed?

        file_url_from_path(Jekyll.sanitized_path(site_source, sass_page.relative_path))
      end

      # The value of the `sourcemap` option chosen by the user.
      #
      # This option controls when sourcemaps shall be generated or not.
      #
      # Returns the value of the `sourcemap`-option chosen by the user or ':always' by default.
      def sourcemap_option
        jekyll_sass_configuration.fetch("sourcemap", :always).to_sym
      end

      # Determines whether a sourcemap shall be generated or not.
      #
      # Returns `true` if a sourcemap shall be generated, `false` otherwise.
      def sourcemap_required?
        return false if associate_page_failed? || sourcemap_option == :never
        return true  if sourcemap_option == :always

        !(sourcemap_option == :development && Jekyll.env != "development")
      end

      def source_map_page
        return if associate_page_failed?

        @source_map_page ||= SourceMapPage.new(sass_page)
      end

      # Returns the directory that source map sources are relative to.
      def sass_source_root
        if associate_page_failed?
          site_source
        else
          Jekyll.sanitized_path(site_source, File.dirname(sass_page.relative_path))
        end
      end

      # Converts file urls in source map to relative paths.
      #
      # Returns processed source map string.
      def process_source_map(source_map)
        map_data = JSON.parse(source_map)
        unless associate_page_failed?
          map_data["file"] = Addressable::URI.encode(sass_page.basename + ".css")
        end
        source_root_url = Addressable::URI.parse(file_url_from_path("#{sass_source_root}/"))
        map_data["sources"].map! do |s|
          s.start_with?("file:") ? Addressable::URI.parse(s).route_from(source_root_url).to_s : s
        end
        JSON.generate(map_data)
      end

      # Adds the source-map to the source-map-page and adds it to `site.pages`.
      def generate_source_map_page(source_map)
        return if associate_page_failed?

        source_map_page.source_map(source_map)
        site.pages << source_map_page
      end

      # Returns a source mapping url for given source-map.
      def source_mapping_url
        return if associate_page_failed?

        Addressable::URI.encode(sass_page.basename + ".css.map")
      end

      def site
        associate_page_failed? ? Jekyll.sites.last : sass_page.site
      end

      def site_source
        site.source
      end

      def file_url_from_path(path)
        Addressable::URI.encode("file://#{path.start_with?("/") ? "" : "/"}#{path}")
      end

      # Returns the value of the `quiet_deps`-option chosen by the user or 'false' by default.
      def quiet_deps_option
        !!jekyll_sass_configuration.fetch("quiet_deps", false)
      end

      # Returns the value of the `verbose`-option chosen by the user or 'false' by default.
      def verbose_option
        !!jekyll_sass_configuration.fetch("verbose", false)
      end
    end
  end
end
