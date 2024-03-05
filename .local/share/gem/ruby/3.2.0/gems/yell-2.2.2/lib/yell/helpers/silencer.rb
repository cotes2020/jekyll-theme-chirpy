module Yell #:nodoc:
  module Helpers #:nodoc:
    module Silencer

      # Set the silence pattern
      def silence( *patterns )
        silencer.add(*patterns)
      end

      def silencer
        @__silencer__
      end


      private

      def reset!
        @__silencer__ = Yell::Silencer.new

        super
      end

      def silence!( *messages )
        @__silencer__.silence!(*messages) if silencer.silence?
      end

    end
  end
end

