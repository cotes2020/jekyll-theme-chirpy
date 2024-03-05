require 'monitor'
require "singleton"

module Yell #:nodoc:

  class LoggerNotFound < StandardError
    def message; "Could not find a Yell::Logger instance with the name '#{super}'"; end
  end

  class Repository
    extend MonitorMixin
    include Singleton

    def initialize
      @loggers = {}
    end

    # Set loggers in the repository
    #
    # @example Set a logger
    #   Yell::Repository[ 'development' ] = Yell::Logger.new :stdout
    #
    # @return [Yell::Logger] The logger instance
    def self.[]=( name, logger )
      synchronize { instance.loggers[name] = logger }
    end

    # Get loggers from the repository
    #
    # @example Get the logger
    #   Yell::Repository[ 'development' ]
    #
    # @raise [Yell::LoggerNotFound] Raised when repository does not have that key
    # @return [Yell::Logger] The logger instance
    def self.[]( name )
      synchronize { instance.__fetch__(name) or raise Yell::LoggerNotFound.new(name) }
    end

    # Get the list of all loggers in the repository
    #
    # @return [Hash] The map of loggers
    def self.loggers
      synchronize { instance.loggers }
    end

    # @private
    def loggers
      @loggers
    end

    # @private
    #
    # Fetch the logger by the given name.
    #
    # If the logger could not be found and has a superclass, it 
    # will attempt to look there. This is important for the 
    # Yell::Loggable module.
    def __fetch__( name )
      logger = loggers[name] || loggers[name.to_s]

      if logger.nil? && name.respond_to?(:superclass)
        return __fetch__(name.superclass)
      end

      logger
    end

  end
end

