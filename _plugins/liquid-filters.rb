module Jekyll
   module ChirpyFilters
    def chomp(input, separator)
      return input.chomp(separator)
    end
  end
end

Liquid::Template.register_filter(Jekyll::ChirpyFilters)
