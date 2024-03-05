# frozen_string_literal: true

module HTMLProofer
  class Attribute
    class Url < HTMLProofer::Attribute
      attr_reader :url, :size

      REMOTE_SCHEMES = ["http", "https"].freeze

      def initialize(runner, link_attribute, base_url: nil, extract_size: false)
        super

        if @raw_attribute.nil?
          @url = nil
        else
          @url = @raw_attribute.delete("\u200b").strip
          @url, @size = @url.split(/\s+/) if extract_size
          @url = Addressable::URI.join(base_url, @url).to_s unless blank?(base_url)
          @url = "" if @url.nil?

          swap_urls!
          clean_url!
        end
      end

      def protocol_relative?
        url.start_with?("//")
      end

      def to_s
        @url
      end

      def known_extension?
        return true if hash_link?
        return true if path.end_with?("/")

        ext = File.extname(path)

        # no extension means we use the assumed one
        return @runner.options[:extensions].include?(@runner.options[:assume_extension]) if blank?(ext)

        @runner.options[:extensions].include?(ext)
      end

      def unknown_extension?
        !known_extension?
      end

      def ignore?
        return true if /^javascript:/.match?(@url)
        return true if ignores_pattern?(@runner.options[:ignore_urls])
      end

      def valid?
        !parts.nil?
      end

      def path?
        !parts.host.nil? && !parts.path.nil?
      end

      def parts
        @parts ||= Addressable::URI.parse(@url)
      rescue URI::Error, Addressable::URI::InvalidURIError
        @parts = nil
      end

      def path
        Addressable::URI.unencode(parts.path) unless parts.nil?
      end

      def hash
        parts&.fragment
      end

      # Does the URL have a hash?
      def hash?
        !blank?(hash)
      end

      def scheme
        parts&.scheme
      end

      def remote?
        REMOTE_SCHEMES.include?(scheme)
      end

      def http?
        scheme == "http"
      end

      def https?
        scheme == "https"
      end

      def non_http_remote?
        !scheme.nil? && !remote?
      end

      def host
        parts&.host
      end

      def domain_path
        (host || "") + path
      end

      def query_values
        parts&.query_values
      end

      # checks if a file exists relative to the current pwd
      def exists?
        return true if base64?

        return @runner.checked_paths[absolute_path] if @runner.checked_paths.key?(absolute_path)

        @runner.checked_paths[absolute_path] = File.exist?(absolute_path)
      end

      def base64?
        /^data:image/.match?(@raw_attribute)
      end

      def absolute_path
        path = file_path || @runner.current_filename

        File.expand_path(path, Dir.pwd)
      end

      def file_path
        return if path.nil? || path.empty?

        path_dot_ext = ""

        path_dot_ext = path + @runner.options[:assume_extension] unless blank?(@runner.options[:assume_extension])

        base = if absolute_path?(path) # path relative to root
          # either overwrite with root_dir; or, if source is directory, use that; or, just get the current file's dirname
          @runner.options[:root_dir] || (File.directory?(@runner.current_source) ? @runner.current_source : File.dirname(@runner.current_source))
        # relative links, path is a file
        elsif File.exist?(File.expand_path(path,
          @runner.current_source)) || File.exist?(File.expand_path(path_dot_ext, @runner.current_source))
          File.dirname(@runner.current_filename)
        # relative links in nested dir, path is a file
        elsif File.exist?(File.join(File.dirname(@runner.current_filename),
          path)) || File.exist?(File.join(File.dirname(@runner.current_filename), path_dot_ext))
          File.dirname(@runner.current_filename)
        # relative link, path is a directory
        else
          @runner.current_filename
        end

        file = File.join(base, path)

        if @runner.options[:assume_extension] && File.file?("#{file}#{@runner.options[:assume_extension]}")
          file = "#{file}#{@runner.options[:assume_extension]}"
        elsif File.directory?(file) && !unslashed_directory?(file) # implicit index support
          file = File.join(file, @runner.options[:directory_index_file])
        end

        file
      end

      def unslashed_directory?(file)
        return false unless File.directory?(file)

        !file.end_with?(File::SEPARATOR) && !follow_location?
      end

      def follow_location?
        @runner.options[:typhoeus] && @runner.options[:typhoeus][:followlocation]
      end

      def absolute_path?(path)
        path.start_with?("/")
      end

      # path is external to the file
      def external?
        !internal?
      end

      def internal?
        relative_link? || internal_absolute_link? || hash_link?
      end

      def internal_absolute_link?
        url.start_with?("/")
      end

      def relative_link?
        return false if remote?

        hash_link? || param_link? || url.start_with?(".") || url =~ /^\S/
      end

      def link_points_to_same_page?
        hash_link || param_link
      end

      def hash_link?
        url.start_with?("#")
      end

      def has_hash?
        url.include?("#")
      end

      def param_link?
        url.start_with?("?")
      end

      def sans_hash
        @url.to_s.sub(/##{hash}/, "")
      end

      # catch any obvious issues, like strings in port numbers
      private def clean_url!
        return if @url =~ /^([!#{Regexp.last_match(0)}-;=?-\[\]_a-z~]|%[0-9a-fA-F]{2})+$/

        @url = Addressable::URI.parse(@url).normalize.to_s
      end

      private def swap_urls!
        return @url if blank?(replacements = @runner.options[:swap_urls])

        replacements.each do |link, replace|
          @url = @url.gsub(link, replace)
        end
      end

      private def ignores_pattern?(links_to_ignore)
        return false unless links_to_ignore.is_a?(Array)

        links_to_ignore.each do |link_to_ignore|
          case link_to_ignore
          when String
            return true if link_to_ignore == @raw_attribute
          when Regexp
            return true if link_to_ignore&.match?(@raw_attribute)
          end
        end

        false
      end
    end
  end
end
