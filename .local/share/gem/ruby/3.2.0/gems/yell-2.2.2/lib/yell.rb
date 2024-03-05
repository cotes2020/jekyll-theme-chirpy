# Copyright (c) 2011-2014 Rudolf Schmidt
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

module Yell #:nodoc:
  # Holds all Yell severities
  Severities = ['DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL', 'UNKNOWN'].freeze

  class << self
    # Creates a new logger instance.
    #
    # Refer to #Yell::Loggger for usage.
    #
    # @return [Yell::Logger] The logger instance
    def new( *args, &block )
      Yell::Logger.new(*args, &block)
    end

    # Shortcut to Yell::Level.new
    #
    # @return [Yell::Level] The level instance
    def level( val = nil )
      Yell::Level.new(val)
    end

    # Shortcut to Yell::Repository[]
    #
    # @return [Yell::Logger] The logger instance
    def []( name )
      Yell::Repository[name]
    end

    # Shortcut to Yell::Repository[]=
    #
    # @return [Yell::Logger] The logger instance
    def []=( name, logger )
      Yell::Repository[name] = logger
    end

    # Shortcut to Yell::Fomatter.new
    #
    # @return [Yell::Formatter] A Yell::Formatter instance
    def format( pattern = nil, date_pattern = nil, &block )
      Yell::Formatter.new(pattern, date_pattern, &block)
    end

    # Loads a config from a YAML file
    #
    # @return [Yell::Logger] The logger instance
    def load!( file )
      Yell.new Yell::Configuration.load!(file)
    end

    # Shortcut to Yell::Adapters.register
    def register( name, klass )
      Yell::Adapters.register(name, klass)
    end

    # @private
    def env
      return ENV['YELL_ENV']  if ENV.key? 'YELL_ENV'
      return ENV['RACK_ENV']  if ENV.key? 'RACK_ENV'
      return ENV['RAILS_ENV'] if ENV.key? 'RAILS_ENV'

      if defined?(Rails)
        Rails.env
      else
        'development'
      end
    end

    # @private
    def __deprecate__( version, message, options = {} ) #:nodoc:
      messages = ["Deprecation Warning (since v#{version}): #{message}" ]
      messages << "  before: #{options[:before]}" if options[:before]
      messages << "  after:  #{options[:after]}" if options[:after]

      __warn__(*messages)
    end

    # @private
    def __warn__( *messages ) #:nodoc:
      $stderr.puts "[Yell] " + messages.join("\n")
    rescue Exception => e
      # do nothing
    end

    # @private
    def __fetch__( hash, *args )
      options = args.last.is_a?(Hash) ? args.pop : {}
      value = args.map { |key| hash.fetch(key.to_sym, hash[key.to_s]) }.compact.first

      value.nil? ? options[:default] : value
    end
  end
end

# helpers
require File.dirname(__FILE__) + '/yell/helpers/base'
require File.dirname(__FILE__) + '/yell/helpers/adapter'
require File.dirname(__FILE__) + '/yell/helpers/formatter'
require File.dirname(__FILE__) + '/yell/helpers/level'
require File.dirname(__FILE__) + '/yell/helpers/tracer'
require File.dirname(__FILE__) + '/yell/helpers/silencer'

# classes
require File.dirname(__FILE__) + '/yell/configuration'
require File.dirname(__FILE__) + '/yell/repository'
require File.dirname(__FILE__) + '/yell/event'
require File.dirname(__FILE__) + '/yell/level'
require File.dirname(__FILE__) + '/yell/formatter'
require File.dirname(__FILE__) + '/yell/silencer'
require File.dirname(__FILE__) + '/yell/adapters'
require File.dirname(__FILE__) + '/yell/logger'

# modules
require File.dirname(__FILE__) + '/yell/loggable'

# core extensions
require File.dirname(__FILE__) + '/core_ext/logger'

# register known adapters
Yell.register :null, Yell::Adapters::Base # adapter that does nothing (for convenience only)
Yell.register :file, Yell::Adapters::File
Yell.register :datefile, Yell::Adapters::Datefile
Yell.register :stdout, Yell::Adapters::Stdout
Yell.register :stderr, Yell::Adapters::Stderr
