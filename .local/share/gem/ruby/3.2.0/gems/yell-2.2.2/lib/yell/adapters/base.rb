require 'monitor'

module Yell #:nodoc:
  module Adapters #:nodoc:

    # This class provides the basic interface for all allowed operations on any 
    # adapter implementation. Other adapters should inherit from it for the methods 
    # used by the {Yell::Logger}.
    #
    # Writing your own adapter is really simple. Inherit from the base class and use 
    # the `setup`, `write` and `close` methods. Yell requires the `write` method to be 
    # specified (`setup` and `close` are optional).
    #
    #
    # The following example shows how to define a basic Adapter to format and print 
    # log events to STDOUT:
    #
    #   class PutsAdapter < Yell::Adapters::Base
    #     include Yell::Formatter::Helpers
    #
    #     setup do |options|
    #       self.format = options[:format]
    #     end
    #
    #     write do |event|
    #       message = format.call(event)
    #
    #       STDOUT.puts message
    #     end
    #   end
    #
    #
    # After the Adapter has been written, we need to register it to Yell:
    #
    #   Yell::Adapters.register :puts, PutsAdapter
    #
    # Now, we can use it like so:
    #
    #   logger = Yell.new :puts
    #   logger.info "Hello World!"
    class Base < Monitor
      include Yell::Helpers::Base
      include Yell::Helpers::Level

      class << self
        # Setup your adapter with this helper method.
        #
        # @example
        #   setup do |options|
        #     @file_handle = File.new( '/dev/null', 'w' )
        #   end
        def setup( &block )
          compile!(:setup!, &block)
        end

        # Define your write method with this helper.
        #
        # @example Printing messages to file
        #   write do |event|
        #     @file_handle.puts event.message
        #   end
        def write( &block )
          compile!(:write!, &block)
        end

        # Define your open method with this helper.
        #
        # @example Open a file handle
        #   open do
        #     @stream = ::File.open( 'test.log', ::File::WRONLY|::File::APPEND|::File::CREAT )
        #   end
        def open( &block )
          compile!(:open!, &block)
        end

        # Define your close method with this helper.
        #
        # @example Closing a file handle
        #   close do
        #     @stream.close
        #   end
        def close( &block )
          compile!(:close!, &block)
        end


        private

        # Pretty funky code block, I know but here is what it basically does:
        #
        # @example
        #   compile! :write! do |event|
        #     puts event.message
        #   end
        #
        #   # Is actually defining the `:write!` instance method with a call to super:
        #
        #   def write!( event )
        #     puts event.method
        #     super
        #   end
        def compile!( name, &block )
          # Get the already defined method
          m = instance_method( name )

          # Create a new method with leading underscore
          define_method("_#{name}", &block)
          _m = instance_method("_#{name}")
          remove_method("_#{name}")

          # Define instance method
          define!(name, _m, m, &block)
        end

        # Define instance method by given name and call the unbound
        # methods in order with provided block.
        def define!( name, _m, m, &block )
          if block.arity == 0
            define_method(name) do
              _m.bind(self).call
              m.bind(self).call
            end
          else
            define_method(name) do |*args|
              _m.bind(self).call(*args)
              m.bind(self).call(*args)
            end
          end
        end
      end


      # Initializes a new Adapter.
      #
      # You should not overload the constructor, use #setup instead.
      def initialize( options = {}, &block )
        super() # init the monitor superclass

        reset!
        setup!(options)

        # eval the given block
        block.arity > 0 ? block.call(self) : instance_eval(&block) if block_given?
      end

      # The main method for calling the adapter.
      #
      # The method receives the log `event` and determines whether to 
      # actually write or not.
      def write( event )
        synchronize { write!(event) } if write?(event)
      rescue Exception => e
        # make sure the adapter is closed and re-raise the exception
        synchronize { close }

        raise(e)
      end

      # Close the adapter (stream, connection, etc).
      #
      # Adapter classes should provide their own implementation 
      # of this method.
      def close
        close!
      end

      # Get a pretty string representation of the adapter, including
      def inspect
        inspection = inspectables.map { |m| "#{m}: #{send(m).inspect}" }
        "#<#{self.class.name} #{inspection * ', '}>"
      end


      private

      # Setup the adapter instance.
      #
      # Adapter classes should provide their own implementation 
      # of this method (if applicable).
      def setup!( options )
        self.level = Yell.__fetch__(options, :level)
      end

      # Perform the actual write.
      #
      # Adapter classes must provide their own implementation 
      # of this method.
      def write!( event )
        # Not implemented
      end

      # Perform the actual open.
      #
      # Adapter classes should provide their own implementation 
      # of this method.
      def open!
        # Not implemented
      end

      # Perform the actual close.
      #
      # Adapter classes should provide their own implementation 
      # of this method.
      def close!
        # Not implemented
      end

      # Determine whether to write at the given severity.
      #
      # @example
      #   write? Yell::Event.new( 'INFO', 'Hello Wold!' )
      #
      # @param [Yell::Event] event The log event
      #
      # @return [Boolean] true or false
      def write?( event )
        level.nil? || level.at?(event.level)
      end

      # Get an array of inspected attributes for the adapter.
      def inspectables
        [:level]
      end

    end

  end
end

