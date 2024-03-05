# frozen_string_literal: true

require "date"
require "json"
require "uri"

module HTMLProofer
  class Cache
    include HTMLProofer::Utils

    CACHE_VERSION = 2

    DEFAULT_STORAGE_DIR = File.join("tmp", ".htmlproofer")
    DEFAULT_CACHE_FILE_NAME = "cache.json"

    URI_REGEXP = URI::DEFAULT_PARSER.make_regexp

    attr_reader :exists, :cache_log, :storage_dir, :cache_file

    def initialize(runner, options)
      @runner = runner
      @logger = @runner.logger

      @cache_datetime = Time.now
      @cache_time = @cache_datetime.to_time

      if blank?(options)
        define_singleton_method(:enabled?) { false }
        define_singleton_method(:external_enabled?) { false }
        define_singleton_method(:internal_enabled?) { false }
      else
        # we still consider the cache as enabled, regardless of the specic timeframes
        define_singleton_method(:enabled?) { true }
        setup_cache!(options)

        @external_timeframe = parsed_timeframe(options[:timeframe][:external])
        define_singleton_method(:external_enabled?) { !@external_timeframe.nil? }
        @internal_timeframe = parsed_timeframe(options[:timeframe][:internal])
        define_singleton_method(:internal_enabled?) { !@internal_timeframe.nil? }
      end
    end

    def parsed_timeframe(timeframe)
      return nil if timeframe.nil?

      time, date = timeframe.match(/(\d+)(\D)/).captures
      time = time.to_i
      case date
      when "M"
        time_ago(time, :months)
      when "w"
        time_ago(time, :weeks)
      when "d"
        time_ago(time, :days)
      when "h"
        time_ago(time, :hours)
      else
        raise ArgumentError, "#{date} is not a valid timeframe!"
      end
    end

    def add_internal(url, metadata, found)
      return unless internal_enabled?

      @cache_log[:internal][url] = { time: @cache_time, metadata: [] } if @cache_log[:internal][url].nil?

      @cache_log[:internal][url][:metadata] << construct_internal_link_metadata(metadata, found)
    end

    def add_external(url, filenames, status_code, msg, found)
      return unless external_enabled?

      clean_url = cleaned_url(url)
      @cache_log[:external][clean_url] =
        { time: @cache_time.to_s, found: found, status_code: status_code, message: msg, metadata: filenames }
    end

    def detect_url_changes(urls_detected, type)
      determine_deletions(urls_detected, type)

      additions = determine_additions(urls_detected, type)

      additions
    end

    def write
      return unless enabled?

      File.write(@cache_file, @cache_log.to_json)
    end

    def retrieve_urls(urls_detected, type)
      # if there are no urls, bail
      return {} if urls_detected.empty?

      urls_detected = urls_detected.transform_keys do |url|
        cleaned_url(url)
      end

      urls_to_check = detect_url_changes(urls_detected, type)

      urls_to_check
    end

    def within_external_timeframe?(time)
      within_timeframe?(time, @external_timeframe)
    end

    def within_internal_timeframe?(time)
      within_timeframe?(time, @internal_timeframe)
    end

    def empty?
      blank?(@cache_log) || (@cache_log[:internal].empty? && @cache_log[:external].empty?)
    end

    def size(type)
      @cache_log[type].size
    end

    private def construct_internal_link_metadata(metadata, found)
      {
        source: metadata[:source],
        filename: metadata[:filename],
        line: metadata[:line],
        base_url: metadata[:base_url],
        found: found,
      }
    end

    # prepare to add new URLs detected
    private def determine_additions(urls_detected, type)
      additions = type == :external ? determine_external_additions(urls_detected) : determine_internal_additions(urls_detected)

      new_link_count = additions.length
      new_link_text = pluralize(new_link_count, "new #{type} link", "new #{type} links")
      @logger.log(:debug, "Adding #{new_link_text} to the cache")

      additions
    end

    private def determine_external_additions(urls_detected)
      urls_detected.reject do |url, _metadata|
        if @cache_log[:external].include?(url)
          found = @cache_log[:external][url][:found] # if this is false, we're trying again
          unless found
            @logger.log(:debug, "Adding #{url} to external cache (not found)")
          end
          found
        else
          @logger.log(:debug, "Adding #{url} to external cache")
          false
        end
      end
    end

    private def determine_internal_additions(urls_detected)
      urls_detected.each_with_object({}) do |(url, detected_metadata), hsh|
        # url is not even in cache
        if @cache_log[:internal][url].nil?
          @logger.log(:debug, "Adding #{url} to internal cache")
          hsh[url] = detected_metadata
          next
        end

        # detect metadata additions
        # NOTE: the time-stamp for the whole url key will not be updated,
        # so that it reflects the earliest time any of the metadata was checked
        cache_metadata = @cache_log[:internal][url][:metadata]
        metadata_additions = detected_metadata.reject do |detected|
          existing_cache_metadata = cache_metadata.find { |cached, _| cached[:filename] == detected[:filename] }
          # cache for this url, from an existing path, exists as found
          found = !existing_cache_metadata.nil? && !existing_cache_metadata.empty? && existing_cache_metadata[:found]
          unless found
            @logger.log(:debug, "Adding #{detected} to internal cache for #{url}")
          end
          found
        end

        if metadata_additions.empty?
          next
        end

        hsh[url] = metadata_additions
        # remove from the cache the detected metadata additions as they correspond to failures to be rechecked
        # (this works assuming the detected url metadata have "found" set to false)
        @cache_log[:internal][url][:metadata] = cache_metadata.difference(metadata_additions)
      end
    end

    # remove from cache URLs that no longer exist
    private def determine_deletions(urls_detected, type)
      deletions = 0

      @cache_log[type].delete_if do |url, cache|
        expired_timeframe = type == :external ? !within_external_timeframe?(cache[:time]) : !within_internal_timeframe?(cache[:time])
        if expired_timeframe
          @logger.log(:debug, "Removing #{url} from #{type} cache (expired timeframe)")
          deletions += 1
          true
        elsif urls_detected.include?(url)
          false
        elsif url_matches_type?(url, type)
          @logger.log(:debug, "Removing #{url} from #{type} cache (not detected anymore)")
          deletions += 1
          true
        end
      end

      del_link_text = pluralize(deletions, "outdated #{type} link", "outdated #{type} links")
      @logger.log(:debug, "Removing #{del_link_text} from the cache")
    end

    private def setup_cache!(options)
      default_structure = {
        version: CACHE_VERSION,
        internal: {},
        external: {},
      }

      @storage_dir = options[:storage_dir] || DEFAULT_STORAGE_DIR

      FileUtils.mkdir_p(storage_dir) unless Dir.exist?(storage_dir)

      cache_file_name = options[:cache_file] || DEFAULT_CACHE_FILE_NAME

      @cache_file = File.join(storage_dir, cache_file_name)

      return (@cache_log = default_structure) unless File.exist?(@cache_file)

      contents = File.read(@cache_file)

      return (@cache_log = default_structure) if blank?(contents)

      log = JSON.parse(contents, symbolize_names: true)

      old_cache = (cache_version = log[:version]).nil?
      @cache_log = if old_cache # previous cache version, create a new one
        default_structure
      elsif cache_version != CACHE_VERSION
      # if cache version is newer...do something
      else
        log[:internal] = log[:internal].transform_keys(&:to_s)
        log[:external] = log[:external].transform_keys(&:to_s)
        log
      end
    end

    # https://github.com/rails/rails/blob/3872bc0e54d32e8bf3a6299b0bfe173d94b072fc/activesupport/lib/active_support/duration.rb#L112-L117
    SECONDS_PER_HOUR   = 3600
    SECONDS_PER_DAY    = 86400
    SECONDS_PER_WEEK   = 604800
    SECONDS_PER_MONTH  = 2629746  # 1/12 of a gregorian year

    private def time_ago(measurement, unit)
      case unit
      when :months
        @cache_datetime - (SECONDS_PER_MONTH * measurement)
      when :weeks
        @cache_datetime - (SECONDS_PER_WEEK * measurement)
      when :days
        @cache_datetime - (SECONDS_PER_DAY * measurement)
      when :hours
        @cache_datetime - Rational(SECONDS_PER_HOUR * measurement)
      end.to_time
    end

    private def url_matches_type?(url, type)
      return true if type == :internal && url !~ URI_REGEXP
      return true if type == :external && url =~ URI_REGEXP
    end

    private def cleaned_url(url)
      cleaned_url = escape_unescape(url)

      return cleaned_url unless cleaned_url.end_with?("/", "#", "?") && cleaned_url.length > 1

      cleaned_url[0..-2]
    end

    private def escape_unescape(url)
      Addressable::URI.parse(url).normalize.to_s
    end

    private def within_timeframe?(current_time, parsed_timeframe)
      return false if current_time.nil? || parsed_timeframe.nil?

      current_time = Time.parse(current_time) if current_time.is_a?(String)
      (parsed_timeframe..@cache_time).cover?(current_time)
    end
  end
end
