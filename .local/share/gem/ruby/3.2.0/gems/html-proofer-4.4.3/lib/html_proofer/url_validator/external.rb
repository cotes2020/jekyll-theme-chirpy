# frozen_string_literal: true

require "typhoeus"
require "uri"

module HTMLProofer
  class UrlValidator
    class External < UrlValidator
      include HTMLProofer::Utils

      attr_reader :external_urls
      attr_writer :before_request

      def initialize(runner, external_urls)
        super(runner)

        @external_urls = external_urls
        @hydra = Typhoeus::Hydra.new(@runner.options[:hydra])
        @before_request = []

        @paths_with_queries = {}
      end

      def validate
        urls_to_check = @cache.external_enabled? ? @runner.load_external_cache : @external_urls
        urls_detected = pluralize(urls_to_check.count, "external link", "external links")
        @logger.log(:info, "Checking #{urls_detected}")

        run_external_link_checker(urls_to_check)

        @failed_checks
      end

      # Proofer runs faster if we pull out all the external URLs and run the checks
      # at the end. Otherwise, we're halting the consuming process for every file during
      # `process_files`.
      #
      # In addition, sorting the list lets libcurl keep connections to the same hosts alive.
      #
      # Finally, we'll first make a HEAD request, rather than GETing all the contents.
      # If the HEAD fails, we'll fall back to GET, as some servers are not configured
      # for HEAD. If we've decided to check for hashes, we must do a GET--HEAD is
      # not available as an option.
      def run_external_link_checker(external_urls)
        # Route log from Typhoeus/Ethon to our own logger
        Ethon.logger = @logger

        external_urls.each_pair do |external_url, metadata|
          url = Attribute::Url.new(@runner, external_url, base_url: nil)

          unless url.valid?
            add_failure(metadata, "#{url} is an invalid URL", 0)
            next
          end

          next unless new_url_query_values?(url)

          method = if @runner.options[:check_external_hash] && url.hash?
            :get
          else
            :head
          end

          queue_request(method, url, metadata)
        end

        @hydra.run
      end

      def queue_request(method, url, filenames)
        opts = @runner.options[:typhoeus].merge(method: method)
        request = Typhoeus::Request.new(url.url, opts)
        @before_request.each do |callback|
          callback.call(request)
        end
        request.on_complete { |response| response_handler(response, url, filenames) }
        @hydra.queue(request)
      end

      def response_handler(response, url, filenames)
        method = response.request.options[:method]
        href = response.request.base_url.to_s
        response_code = response.code
        response.body.delete!("\x00")

        @logger.log(:debug, "Received a #{response_code} for #{href}")

        return if @runner.options[:ignore_status_codes].include?(response_code)

        if response_code.between?(200, 299)
          @cache.add_external(href, filenames, response_code, "OK", true) unless check_hash_in_2xx_response(href, url,
            response, filenames)
        elsif response.timed_out?
          handle_timeout(href, filenames, response_code)
        elsif response_code.zero?
          handle_connection_failure(href, filenames, response_code, response.status_message)
        elsif method == :head # some servers don't support HEAD
          queue_request(:get, url, filenames)
        else
          return if @runner.options[:only_4xx] && !response_code.between?(400, 499)

          # Received a non-successful http response.
          status_message = blank?(response.status_message) ? "" : ": #{response.status_message}"
          msg = "External link #{href} failed#{status_message}"
          add_failure(filenames, msg, response_code)
          @cache.add_external(href, filenames, response_code, msg, false)
        end
      end

      # Even though the response was a success, we may have been asked to check
      # if the hash on the URL exists on the page
      def check_hash_in_2xx_response(href, url, response, filenames)
        return false if @runner.options[:only_4xx]
        return false unless @runner.options[:check_external_hash]
        return false unless url.hash?

        hash = url.hash

        body_doc = create_nokogiri(response.body)

        unencoded_hash = Addressable::URI.unescape(hash)
        xpath = [%(//*[@name="#{hash}"]|/*[@name="#{unencoded_hash}"]|//*[@id="#{hash}"]|//*[@id="#{unencoded_hash}"])]
        # user-content is a special addition by GitHub.
        if url.host =~ /github\.com/i
          xpath << [%(//*[@name="user-content-#{hash}"]|//*[@id="user-content-#{hash}"])]
          # when linking to a file on GitHub, like #L12-L34, only the first "L" portion
          # will be identified as a linkable portion
          xpath << [%(//td[@id="#{Regexp.last_match[1]}"])] if hash =~ /\A(L\d)+/
        end

        return unless body_doc.xpath(xpath.join("|")).empty?

        msg = "External link #{href} failed: #{url.sans_hash} exists, but the hash '#{hash}' does not"
        add_failure(filenames, msg, response.code)
        @cache.add_external(href, filenames, response.code, msg, false)
        true
      end

      def handle_timeout(href, filenames, response_code)
        msg = "External link #{href} failed: got a time out (response code #{response_code})"
        @cache.add_external(href, filenames, 0, msg, false)
        return if @runner.options[:only_4xx]

        add_failure(filenames, msg, response_code)
      end

      def handle_connection_failure(href, metadata, response_code, status_message)
        msgs = [<<~MSG,
          External link #{href} failed with something very wrong.
          It's possible libcurl couldn't connect to the server, or perhaps the request timed out.
          Sometimes, making too many requests at once also breaks things.
        MSG
        ]

        msgs << "Either way, the return message from the server is: #{status_message}" unless blank?(status_message)

        msg = msgs.join("\n").chomp

        @cache.add_external(href, metadata, 0, msg, false)
        return if @runner.options[:only_4xx]

        add_failure(metadata, msg, response_code)
      end

      def add_failure(metadata, description, status = nil)
        if blank?(metadata) # possible if we're checking an array of links
          @failed_checks << Failure.new("", "Links > External", description, status: status)
        else
          metadata.each do |m|
            @failed_checks << Failure.new(m[:filename], "Links > External", description, line: m[:line], status: status)
          end
        end
      end

      # remember queries we've seen, ignore future ones
      private def new_url_query_values?(url)
        return true if (query_values = url.query_values).nil?

        queries = query_values.keys.join("-")
        domain_path = url.domain_path
        if @paths_with_queries[domain_path].nil?
          @paths_with_queries[domain_path] = [queries]
          true
        elsif !@paths_with_queries[domain_path].include?(queries)
          @paths_with_queries[domain_path] << queries
          true
        else
          false
        end
      end
    end
  end
end
