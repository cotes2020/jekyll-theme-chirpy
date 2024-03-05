# frozen_string_literal: true

module Sass
  class Compiler
    # The {Varint} module.
    #
    # It reads and writes varints.
    module Varint
      module_function

      def length(value)
        return 1 if value < 128

        (value.bit_length + 6) / 7
      end

      def read(readable)
        value = bits = 0
        loop do
          byte = readable.readbyte
          value |= (byte & 0x7f) << bits
          bits += 7
          break if byte < 0x80
        end
        value
      end

      def write(writeable, value)
        until value < 0x80
          writeable << ((value & 0x7f) | 0x80)
          value >>= 7
        end
        writeable << value
      end
    end

    private_constant :Varint
  end
end
