# frozen_string_literal: true

module HTMLProofer
  class Check
    class Links < HTMLProofer::Check
      def run
        @html.css("a, link").each do |node|
          @link = create_element(node)

          next if @link.ignore?

          if !allow_hash_href? && @link.node["href"] == "#"
            add_failure("linking to internal hash #, which points to nowhere", line: @link.line, content: @link.content)
            next
          end

          # is there even an href?
          if blank?(@link.url.raw_attribute)
            next if allow_missing_href?

            add_failure("'#{@link.node.name}' tag is missing a reference", line: @link.line, content: @link.content)
            next
          end

          # is it even a valid URL?
          unless @link.url.valid?
            add_failure("#{@link.href} is an invalid URL", line: @link.line, content: @link.content)
            next
          end

          if @link.url.protocol_relative?
            add_failure("#{@link.url} is a protocol-relative URL, use explicit https:// instead",
              line: @link.line, content: @link.content)
            next
          end

          check_schemes

          # intentionally down here because we still want valid? & missing_href? to execute
          next if @link.url.non_http_remote?

          if !@link.url.internal? && @link.url.remote?
            check_sri if @runner.check_sri? && @link.link_tag?

            # we need to skip these for now; although the domain main be valid,
            # curl/Typheous inaccurately return 404s for some links. cc https://git.io/vyCFx
            next if @link.node["rel"] == "dns-prefetch"

            unless @link.url.path?
              add_failure("#{@link.url.raw_attribute} is an invalid URL", line: @link.line, content: @link.content)
              next
            end

            add_to_external_urls(@link.url, @link.line)
          elsif @link.url.internal?
            # does the local directory have a trailing slash?
            if @link.url.unslashed_directory?(@link.url.absolute_path)
              add_failure("internally linking to a directory #{@link.url.raw_attribute} without trailing slash",
                line: @link.line, content: @link.content)
              next
            end

            add_to_internal_urls(@link.url, @link.line)
          end
        end
      end

      def allow_missing_href?
        @runner.options[:allow_missing_href]
      end

      def allow_hash_href?
        @runner.options[:allow_hash_href]
      end

      def check_schemes
        case @link.url.scheme
        when "mailto"
          handle_mailto
        when "tel"
          handle_tel
        when "http"
          return unless @runner.options[:enforce_https]

          add_failure("#{@link.url.raw_attribute} is not an HTTPS link", line: @link.line, content: @link.content)
        end
      end

      def handle_mailto
        if @link.url.path.empty?
          add_failure("#{@link.url.raw_attribute} contains no email address", line: @link.line,
            content: @link.content) unless ignore_empty_mailto?
        elsif !/#{URI::MailTo::EMAIL_REGEXP}/o.match?(@link.url.path)
          add_failure("#{@link.url.raw_attribute} contains an invalid email address", line: @link.line,
            content: @link.content)
        end
      end

      def handle_tel
        add_failure("#{@link.url.raw_attribute} contains no phone number", line: @link.line,
          content: @link.content) if @link.url.path.empty?
      end

      def ignore_empty_mailto?
        @runner.options[:ignore_empty_mailto]
      end

      # Allowed elements from Subresource Integrity specification
      # https://w3c.github.io/webappsec-subresource-integrity/#link-element-for-stylesheets
      SRI_REL_TYPES = %(stylesheet)

      def check_sri
        return unless SRI_REL_TYPES.include?(@link.node["rel"])

        if blank?(@link.node["integrity"]) && blank?(@link.node["crossorigin"])
          add_failure("SRI and CORS not provided in: #{@link.url.raw_attribute}", line: @link.line,
            content: @link.content)
        elsif blank?(@link.node["integrity"])
          add_failure("Integrity is missing in: #{@link.url.raw_attribute}", line: @link.line, content: @link.content)
        elsif blank?(@link.node["crossorigin"])
          add_failure("CORS not provided for external resource in: #{@link.link.url.raw_attribute}", line: @link.line,
            content: @link.content)
        end
      end

      private def source_tag?
        @link.node.name == "source"
      end

      private def anchor_tag?
        @link.node.name == "a"
      end
    end
  end
end
