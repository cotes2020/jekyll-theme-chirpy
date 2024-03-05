# frozen_string_literal: true

module Sass
  # A namespace for built-in Loggers.
  #
  # @see https://sass-lang.com/documentation/js-api/modules/logger/
  module Logger
    module_function

    # A Logger that silently ignores all warnings and debug messages.
    #
    # @see https://sass-lang.com/documentation/js-api/variables/logger.silent/
    def silent
      Silent
    end

    # A Logger that silently ignores all warnings and debug messages.
    module Silent
      module_function

      def warn(message, deprecation: false, span: nil, stack: nil); end

      def debug(message, span: nil); end
    end

    private_constant :Silent
  end
end
