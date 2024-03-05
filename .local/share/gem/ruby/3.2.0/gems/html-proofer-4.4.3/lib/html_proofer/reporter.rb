# frozen_string_literal: true

module HTMLProofer
  class Reporter
    include HTMLProofer::Utils

    attr_reader :failures

    def initialize(logger: nil)
      @logger = logger
    end

    def failures=(failures)
      @failures = failures.group_by(&:check_name) \
        .transform_values { |issues| issues.sort_by { |issue| [issue.path, issue.line] } } \
        .sort
    end

    def report
      raise NotImplementedError, "HTMLProofer::Reporter subclasses must implement #report"
    end
  end
end
