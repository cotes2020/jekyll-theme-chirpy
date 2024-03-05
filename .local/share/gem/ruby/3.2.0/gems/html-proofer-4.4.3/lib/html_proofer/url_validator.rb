# frozen_string_literal: true

module HTMLProofer
  class UrlValidator
    include HTMLProofer::Utils

    def initialize(runner)
      @runner = runner

      @cache = @runner.cache
      @logger = @runner.logger

      @failed_checks = []
    end
  end
end
