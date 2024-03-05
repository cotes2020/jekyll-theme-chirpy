# frozen_string_literal: true

module HTMLProofer
  class Runner
    include HTMLProofer::Utils

    attr_reader :options, :cache, :logger, :internal_urls, :external_urls, :checked_paths, :current_check
    attr_accessor :current_filename, :current_source, :reporter

    URL_TYPES = [:external, :internal].freeze

    def initialize(src, opts = {})
      @options = HTMLProofer::Configuration.generate_defaults(opts)

      @type = @options.delete(:type)
      @source = src

      @logger = HTMLProofer::Log.new(@options[:log_level])
      @cache = Cache.new(self, @options[:cache])

      @external_urls = {}
      @internal_urls = {}
      @failures = []

      @before_request = []

      @checked_paths = {}

      @current_check = nil
      @current_source = nil
      @current_filename = nil

      @reporter = Reporter::Cli.new(logger: @logger)
    end

    def run
      check_text = pluralize(checks.length, "check", "checks")

      if @type == :links
        @logger.log(:info, "Running #{check_text} (#{format_checks_list(checks)}) on #{@source} ... \n\n")
        check_list_of_links unless @options[:disable_external]
      else
        @logger.log(:info,
          "Running #{check_text} (#{format_checks_list(checks)}) in #{@source} on *#{@options[:extensions].join(", ")} files...\n\n")

        check_files
        @logger.log(:info, "Ran on #{pluralize(files.length, "file", "files")}!\n\n")
      end

      @cache.write

      @reporter.failures = @failures

      if @failures.empty?
        @logger.log(:info, "HTML-Proofer finished successfully.")
      else
        @failures.uniq!
        report_failed_checks
      end
    end

    def check_list_of_links
      @external_urls = @source.uniq.each_with_object({}) do |link, hash|
        url = Attribute::Url.new(self, link, base_url: nil).to_s

        hash[url] = []
      end

      validate_external_urls
    end

    # Walks over each implemented check and runs them on the files, in parallel.
    # Sends the collected external URLs to Typhoeus for batch processing.
    def check_files
      process_files.each do |result|
        URL_TYPES.each do |url_type|
          type = :"#{url_type}_urls"
          ivar_name = "@#{type}"
          ivar = instance_variable_get(ivar_name)

          if ivar.empty?
            instance_variable_set(ivar_name, result[type])
          else
            result[type].each do |url, metadata|
              ivar[url] = [] if ivar[url].nil?
              ivar[url].concat(metadata)
            end
          end
        end
        @failures.concat(result[:failures])
      end

      validate_external_urls unless @options[:disable_external]

      validate_internal_urls
    end

    # Walks over each implemented check and runs them on the files, in parallel.
    def process_files
      if @options[:parallel][:enable]
        Parallel.map(files, @options[:parallel]) { |file| load_file(file[:path], file[:source]) }
      else
        files.map do |file|
          load_file(file[:path], file[:source])
        end
      end
    end

    def load_file(path, source)
      @html = create_nokogiri(path)
      check_parsed(path, source)
    end

    # Collects any external URLs found in a directory of files. Also collectes
    # every failed test from process_files.
    def check_parsed(path, source)
      result = { internal_urls: {}, external_urls: {}, failures: [] }

      checks.each do |klass|
        @current_source = source
        @current_filename = path

        check = Object.const_get(klass).new(self, @html)
        @logger.log(:debug, "Running #{check.short_name} in #{path}")

        @current_check = check

        check.run

        result[:external_urls].merge!(check.external_urls) { |_key, old, current| old.concat(current) }
        result[:internal_urls].merge!(check.internal_urls) { |_key, old, current| old.concat(current) }
        result[:failures].concat(check.failures)
      end
      result
    end

    def validate_external_urls
      external_url_validator = HTMLProofer::UrlValidator::External.new(self, @external_urls)
      external_url_validator.before_request = @before_request
      @failures.concat(external_url_validator.validate)
    end

    def validate_internal_urls
      internal_link_validator = HTMLProofer::UrlValidator::Internal.new(self, @internal_urls)
      @failures.concat(internal_link_validator.validate)
    end

    def files
      @files ||= if @type == :directory
        @source.map do |src|
          pattern = File.join(src, "**", "*{#{@options[:extensions].join(",")}}")
          Dir.glob(pattern).select do |f|
            File.file?(f) && !ignore_file?(f)
          end.map { |f| { source: src, path: f } }
        end.flatten
      elsif @type == :file && @options[:extensions].include?(File.extname(@source))
        [@source].reject { |f| ignore_file?(f) }.map { |f| { source: f, path: f } }
      else
        []
      end
    end

    def ignore_file?(file)
      @options[:ignore_files].each do |pattern|
        return true if pattern.is_a?(String) && pattern == file
        return true if pattern.is_a?(Regexp) && pattern =~ file
      end

      false
    end

    def check_sri?
      @options[:check_sri]
    end

    def enforce_https?
      @options[:enforce_https]
    end

    def checks
      return @checks if defined?(@checks) && !@checks.nil?

      return (@checks = ["LinkCheck"]) if @type == :links

      @checks = HTMLProofer::Check.subchecks(@options).map(&:name)

      @checks
    end

    def failed_checks
      @reporter.failures.flatten.select { |f| f.is_a?(Failure) }
    end

    def report_failed_checks
      @reporter.report

      failure_text = pluralize(@failures.length, "failure", "failures")
      @logger.log(:fatal, "\nHTML-Proofer found #{failure_text}!")
      exit(1)
    end

    # Set before_request callback.
    #
    # @example Set before_request.
    #   request.before_request { |request| p "yay" }
    #
    # @param [ Block ] block The block to execute.
    #
    # @yield [ Typhoeus::Request ]
    #
    # @return [ Array<Block> ] All before_request blocks.
    def before_request(&block)
      @before_request ||= []
      @before_request << block if block
      @before_request
    end

    def load_internal_cache
      load_cache(:internal)
    end

    def load_external_cache
      load_cache(:external)
    end

    private def load_cache(type)
      ivar = instance_variable_get("@#{type}_urls")

      existing_urls_count = @cache.size(type)
      cache_text = pluralize(existing_urls_count, "#{type} link", "#{type} links")
      @logger.log(:debug, "Found #{cache_text} in the cache")

      urls_to_check = @cache.retrieve_urls(ivar, type)

      urls_to_check
    end

    private def format_checks_list(checks)
      checks.map do |check|
        check.sub(/HTMLProofer::Check::/, "")
      end.join(", ")
    end
  end
end
