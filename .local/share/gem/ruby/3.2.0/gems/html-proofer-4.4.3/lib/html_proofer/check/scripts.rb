# frozen_string_literal: true

module HTMLProofer
  class Check
    class Scripts < HTMLProofer::Check
      def run
        @html.css("script").each do |node|
          @script = create_element(node)

          next if @script.ignore?
          next unless @script.content.strip.empty?

          # does the script exist?
          if missing_src?
            add_failure("script is empty and has no src attribute", line: @script.line, content: @script.content)
          elsif @script.url.protocol_relative?
            add_failure("script link #{@script.url} is a protocol-relative URL, use explicit https:// instead",
              line: @script.line, content: @script.content)
          elsif @script.url.remote?
            add_to_external_urls(@script.url, @script.line)
            check_sri if @runner.check_sri?
          elsif !@script.url.exists?
            add_failure("internal script reference #{@script.src} does not exist", line: @script.line,
              content: @script.content)
          end
        end

        external_urls
      end

      def missing_src?
        @script.node["src"].nil?
      end

      def check_sri
        if blank?(@script.node["integrity"]) && blank?(@script.node["crossorigin"])
          add_failure("SRI and CORS not provided in: #{@script.url.raw_attribute}", line: @script.line,
            content: @script.content)
        elsif blank?(@script.node["integrity"])
          add_failure("Integrity is missing in: #{@script.url.raw_attribute}", line: @script.line,
            content: @script.content)
        elsif blank?(@script.node["crossorigin"])
          add_failure("CORS not provided for external resource in: #{@script.url.raw_attribute}", line: @script.line,
            content: @script.content)
        end
      end
    end
  end
end
