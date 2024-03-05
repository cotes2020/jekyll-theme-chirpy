module Yell #:nodoc:
  # Include this module to add a logger to any class.
  #
  # When including this module, your class will have a :logger instance method 
  # available. Before you can use it, you will need to define a Yell logger and 
  # provide it with the name of your class.
  #
  # @example
  #   Yell.new :stdout, name: 'Foo'
  #
  #   class Foo
  #     include Yell::Loggable
  #   end
  #
  #   Foo.new.logger.info "Hello World"
  module Loggable
    def self.included(base)
      base.extend(ClassMethods)
    end

    module ClassMethods
      def logger
        Yell[self]
      end
    end

    def logger
      self.class.logger
    end
  end
end
