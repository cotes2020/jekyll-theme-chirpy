# frozen_string_literal: true

module HTMLProofer
  class Reporter
    class Cli < HTMLProofer::Reporter
      def report
        msg = failures.each_with_object([]) do |(check_name, failures), arr|
          str = ["For the #{check_name} check, the following failures were found:\n"]

          failures.each do |failure|
            path_str = blank?(failure.path) ? "" : "At #{failure.path}"

            line_str = failure.line.nil? ? "" : ":#{failure.line}"

            path_and_line = "#{path_str}#{line_str}"
            path_and_line = blank?(path_and_line) ? "" : "* #{path_and_line}:\n\n"

            status_str = failure.status.nil? ? "" : " (status code #{failure.status})"

            indent = blank?(path_and_line) ? "* " : "  "
            str << <<~MSG
              #{path_and_line}#{indent}#{failure.description}#{status_str}
            MSG
          end

          arr << str.join("\n")
        end

        @logger.log(:error, msg.join("\n"))
      end
    end
  end
end
