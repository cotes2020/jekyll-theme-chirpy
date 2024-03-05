# frozen_string_literal: true

module Sass
  class Compiler
    class Host
      # The {Structifier} module.
      #
      # It converts {::Hash} to {Struct}-like objects.
      module Structifier
        module_function

        def to_struct(obj, *symbols)
          return obj unless obj.is_a?(Hash)

          struct = Object.new
          symbols.each do |key|
            next unless obj.key?(key)

            value = obj[key]
            if value.respond_to?(:call)
              struct.define_singleton_method key do |*args, **kwargs|
                value.call(*args, **kwargs)
              end
            else
              struct.define_singleton_method key do
                value
              end
            end
          end
          struct
        end
      end

      private_constant :Structifier
    end
  end
end
