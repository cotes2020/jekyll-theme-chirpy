# frozen_string_literal: true

require_relative 'compiler'

# The Sass module.
#
# This communicates with Embedded Dart Sass using the Embedded Sass protocol.
#
# @example
#   Sass.compile('style.scss')
#
# @example
#   Sass.compile_string('h1 { font-size: 40px; }')
module Sass
  @compiler = nil
  @mutex = Mutex.new

  # rubocop:disable Layout/LineLength
  class << self
    # Compiles the Sass file at +path+ to CSS.
    # @overload compile(path, load_paths: [], charset: true, source_map: false, source_map_include_sources: false, style: :expanded, functions: {}, importers: [], alert_ascii: false, alert_color: nil, logger: nil, quiet_deps: false, verbose: false)
    # @param (see Compiler#compile)
    # @return (see Compiler#compile)
    # @raise (see Compiler#compile)
    # @see Compiler#compile
    def compile(...)
      compiler.compile(...)
    end

    # Compiles a stylesheet whose contents is +source+ to CSS.
    # @overload compile_string(source, importer: nil, load_paths: [], syntax: :scss, url: nil, charset: true, source_map: false, source_map_include_sources: false, style: :expanded, functions: {}, importers: [], alert_ascii: false, alert_color: nil, logger: nil, quiet_deps: false, verbose: false)
    # @param (see Compiler#compile_string)
    # @return (see Compiler#compile_string)
    # @raise (see Compiler#compile_string)
    # @see Compiler#compile_string
    def compile_string(...)
      compiler.compile_string(...)
    end

    # @param (see Compiler#info)
    # @return (see Compiler#info)
    # @raise (see Compiler#info)
    # @see Compiler#info
    def info
      compiler.info
    end

    private

    def compiler
      return @compiler if @compiler

      @mutex.synchronize do
        return @compiler if @compiler

        compiler = Class.new(Compiler) do
          def initialize
            @channel = Compiler.const_get(:Channel).new(Class.new(Compiler.const_get(:Dispatcher)) do
              def initialize
                super

                idle_timeout = 10
                @last_accessed_time = current_time

                Thread.new do
                  Thread.current.name = 'sass-embedded-process-reaper'
                  duration = idle_timeout
                  loop do
                    sleep(duration.negative? ? idle_timeout : duration)
                    break if @mutex.synchronize do
                      raise Errno::EBUSY if _closed?

                      duration = idle_timeout - (current_time - @last_accessed_time)
                      duration.negative? && _idle? && _close
                    end
                  end
                  close
                rescue Errno::EBUSY
                  # do nothing
                end
              end

              private

              def _idle
                super

                @last_accessed_time = current_time
              end

              def current_time
                Process.clock_gettime(Process::CLOCK_MONOTONIC)
              end
            end)
          end
        end.new

        at_exit do
          compiler.close
        end

        @compiler = compiler
      end
    end
  end
  # rubocop:enable Layout/LineLength
end
