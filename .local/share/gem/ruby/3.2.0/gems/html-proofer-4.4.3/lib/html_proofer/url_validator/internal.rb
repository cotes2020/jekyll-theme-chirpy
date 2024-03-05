# frozen_string_literal: true

module HTMLProofer
  class UrlValidator
    class Internal < UrlValidator
      attr_reader :internal_urls

      def initialize(runner, internal_urls)
        super(runner)

        @internal_urls = internal_urls
      end

      def validate
        urls_to_check = @cache.internal_enabled? ? @runner.load_internal_cache : @internal_urls
        urls_detected = pluralize(urls_to_check.count, "internal link", "internal links")
        @logger.log(:info, "Checking #{urls_detected}")

        run_internal_link_checker(urls_to_check)

        @failed_checks
      end

      def run_internal_link_checker(links)
        # collect urls and metadata for hashes to be checked in the same target file
        file_paths_hashes_to_check = {}
        to_add = []
        links.each_with_index do |(link, matched_files), i|
          matched_count_to_log = pluralize(matched_files.count, "reference", "references")
          @logger.log(:debug, "(#{i + 1} / #{links.count}) Internal link #{link}: Checking #{matched_count_to_log}")
          matched_files.each do |metadata|
            url = HTMLProofer::Attribute::Url.new(@runner, link, base_url: metadata[:base_url])

            @runner.current_source = metadata[:source]
            @runner.current_filename = metadata[:filename]

            target_file_path = url.absolute_path
            unless file_exists?(target_file_path)
              @failed_checks << Failure.new(@runner.current_filename, "Links > Internal",
                "internally linking to #{url}, which does not exist", line: metadata[:line], status: nil, content: nil)
              to_add << [url, metadata, false]
              next
            end

            hash_exists = hash_exists_for_url?(url)
            if hash_exists.nil?
              # the hash needs to be checked in the target file, we collect the url and metadata
              unless file_paths_hashes_to_check.key?(target_file_path)
                file_paths_hashes_to_check[target_file_path] = {}
              end
              unless file_paths_hashes_to_check[target_file_path].key?(url.hash)
                file_paths_hashes_to_check[target_file_path][url.hash] = []
              end
              file_paths_hashes_to_check[target_file_path][url.hash] << [url, metadata]
              next
            end
            unless hash_exists
              @failed_checks << Failure.new(@runner.current_filename, "Links > Internal",
                "internally linking to #{url}; the file exists, but the hash '#{url.hash}' does not", line: metadata[:line], status: nil, content: nil)
              to_add << [url, metadata, false]
              next
            end

            to_add << [url, metadata, true]
          end
        end

        # check hashes by target file
        @logger.log(:info, "Checking internal link hashes in #{pluralize(file_paths_hashes_to_check.count, "file", "files")}")
        file_paths_hashes_to_check.each_with_index do |(file_path, hashes_to_check), i|
          hash_count_to_log = pluralize(hashes_to_check.count, "hash", "hashes")
          @logger.log(:debug, "(#{i + 1} / #{file_paths_hashes_to_check.count}) Checking #{hash_count_to_log} in #{file_path}")
          html = create_nokogiri(file_path)
          hashes_to_check.each_pair do |href_hash, url_metadata|
            exists = hash_exists_in_html?(href_hash, html)
            url_metadata.each do |(url, metadata)|
              unless exists
                @failed_checks << Failure.new(metadata[:filename], "Links > Internal",
                  "internally linking to #{url}; the file exists, but the hash '#{href_hash}' does not", line: metadata[:line], status: nil, content: nil)
              end
              to_add << [url, metadata, exists]
            end
          end
        end

        # adding directly to the cache above results in an endless loop
        to_add.each do |(url, metadata, exists)|
          @cache.add_internal(url.to_s, metadata, exists)
        end

        @failed_checks
      end

      private def file_exists?(absolute_path)
        return @runner.checked_paths[absolute_path] if @runner.checked_paths.key?(absolute_path)

        @runner.checked_paths[absolute_path] = File.exist?(absolute_path)
      end

      # verify the hash w/o just based on the URL, w/o looking at the target file
      # => returns nil if the has could not be verified
      private def hash_exists_for_url?(url)
        href_hash = url.hash
        return true if blank?(href_hash)
        return true unless @runner.options[:check_internal_hash]

        # prevents searching files we didn't ask about
        return false unless url.known_extension?
        return false unless url.has_hash?

        decoded_href_hash = Addressable::URI.unescape(href_hash)
        fragment_ids = [href_hash, decoded_href_hash]
        # https://www.w3.org/TR/html5/single-page.html#scroll-to-fragid
        return true if fragment_ids.include?("top")

        nil
      end

      private def hash_exists_in_html?(href_hash, html)
        decoded_href_hash = Addressable::URI.unescape(href_hash)
        fragment_ids = [href_hash, decoded_href_hash]
        !find_fragments(fragment_ids, html).empty?
      end

      private def find_fragments(fragment_ids, html)
        xpaths = fragment_ids.uniq.flat_map do |frag_id|
          escaped_frag_id = "'#{frag_id.split("'").join("', \"'\", '")}', ''"
          [
            "//*[case_sensitive_equals(@id, concat(#{escaped_frag_id}))]",
            "//*[case_sensitive_equals(@name, concat(#{escaped_frag_id}))]",
          ]
        end
        xpaths << XpathFunctions.new

        html.xpath(*xpaths)
      end
    end
  end
end
