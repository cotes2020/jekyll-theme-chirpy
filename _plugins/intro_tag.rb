module Jekyll
    class IntroTag < Liquid::Tag
      def initialize(tag_name, file, tokens)
        super
        @file = file.strip
      end
  
      def render(context)
        intro_path = File.join(context.registers[:site].config["source"], '_intro', @file)
        if File.exist?(intro_path)
            content = File.read(intro_path)
            site = context.registers[:site]
            converter = site.find_converter_instance(::Jekyll::Converters::Markdown)
            converter.convert(content)
        else
            "File #{@file} not found"
        end
      end
    end
  end
  
  Liquid::Template.register_tag('intro', Jekyll::IntroTag)
  