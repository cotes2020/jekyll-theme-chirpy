module Yell #:nodoc:
  # AdapterNotFound is raised whenever you want to instantiate an 
  # adapter that does not exist.
  class AdapterNotFound < StandardError; end

  # This module provides the interface to attaching adapters to
  # the logger. You should not have to call the corresponding classes
  # directly.
  module Adapters
    class Collection
      def initialize( options = {} )
        @options = options
        @collection = []
      end

      def add( type = :file, *args, &block )
        options = [@options, *args].inject(Hash.new) do |h, c|
          h.merge( [String, Pathname].include?(c.class) ? {:filename => c} : c  )
        end

        # remove possible :null adapters
        @collection.shift if @collection.first.instance_of?(Yell::Adapters::Base)

        new_adapter = Yell::Adapters.new(type, options, &block)
        @collection.push(new_adapter)

        new_adapter
      end

      def empty?
        @collection.empty?
      end

      # @private
      def write( event )
        @collection.each { |c| c.write(event) }
        true
      end

      # @private
      def close
        @collection.each { |c| c.close }
      end
    end

    # holds the list of known adapters
    @adapters = {}

    # Register your own adapter here
    #
    # @example
    #   Yell::Adapters.register( :myadapter, MyAdapter )
    def self.register( name, klass )
      @adapters[name.to_sym] = klass
    end

    # Returns an instance of the given processor type.
    #
    # @example A simple file adapter
    #   Yell::Adapters.new( :file )
    def self.new( type, options = {}, &block )
      return type if type.is_a?(Yell::Adapters::Base)

      adapter = case type
      when STDOUT then @adapters[:stdout]
      when STDERR then @adapters[:stderr]
      else @adapters[type.to_sym]
      end

      raise AdapterNotFound.new(type) if adapter.nil?
      adapter.new(options, &block)
    end
  end
end

# Base for all adapters
require File.dirname(__FILE__) + '/adapters/base'

# IO based adapters
require File.dirname(__FILE__) + '/adapters/io'
require File.dirname(__FILE__) + '/adapters/streams'
require File.dirname(__FILE__) + '/adapters/file'
require File.dirname(__FILE__) + '/adapters/datefile'

