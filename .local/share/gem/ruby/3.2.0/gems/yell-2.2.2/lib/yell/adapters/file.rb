module Yell #:nodoc:
  module Adapters #:nodoc:

    # The +File+ adapter is the most basic. As one would expect, it's used 
    # for logging into files.
    class File < Yell::Adapters::Io

      private

      # @overload setup!( options )
      def setup!( options )
        @filename = ::File.expand_path(Yell.__fetch__(options, :filename, default: default_filename))

        super
      end

      # @overload open!
      def open!
        @stream = ::File.open(@filename, ::File::WRONLY|::File::APPEND|::File::CREAT)

        super
      end

      def default_filename #:nodoc:
        logdir = ::File.expand_path("log")

        ::File.expand_path(::File.directory?(logdir) ? "#{logdir}/#{Yell.env}.log" : "#{Yell.env}.log")
      end

    end

  end
end

