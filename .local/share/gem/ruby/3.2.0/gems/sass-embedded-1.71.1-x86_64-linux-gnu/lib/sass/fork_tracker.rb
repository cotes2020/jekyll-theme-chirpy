# frozen_string_literal: true

module Sass
  # The {ForkTracker} module.
  #
  # It tracks objects that need to be closed after `Process.fork`.
  module ForkTracker
    HASH = {}.compare_by_identity

    MUTEX = Mutex.new

    private_constant :HASH, :MUTEX

    module_function

    def add(obj)
      MUTEX.synchronize do
        HASH[obj] = true
      end
    end

    def delete(obj)
      MUTEX.synchronize do
        HASH.delete(obj)
      end
    end

    def each(...)
      MUTEX.synchronize do
        HASH.keys
      end.each(...)
    end

    # The {CoreExt} module.
    #
    # It closes objects after `Process.fork`.
    module CoreExt
      def _fork
        pid = super
        ForkTracker.each(&:close) if pid.zero?
        pid
      end
    end

    private_constant :CoreExt

    Process.singleton_class.prepend(CoreExt) if Process.respond_to?(:_fork)
  end

  private_constant :ForkTracker
end
