# Jekyll Lightbox Plugin
#
# Bart Vandeweerdt | www.appfoundry.be
#
# Example usage: {% lightbox images/appfoundry.png --thumb="images/thumbs/appfoundry.png" --data="some data" --title="some title" --alt="some alt" --img-style="css styling" --class="yourclass"%}
module Jekyll
  class LightboxTag < Liquid::Tag

    def initialize(tag_name, text, token)
      super

      # The path to our image
      @path = Liquid::Template.parse(
        # Regex: split on first whitespace character while allowing double quoting for surrounding spaces in a file path
        text.split(/\s(?=(?:[^"]|"[^"]*")*$)/)[0].strip
      ).render(@context)

      # Defaults
      @title = ''
      @alt = ''
      @img_style = ''
      @class = ''
      @data = ''
      @thumb = @path

      # Parse Options
      if text =~ /--title="([^"]*)"/i
        @title = text.match(/--title="([^"]*)"/i)[1]
      end
      if text =~ /--alt="([^"]*)"/i
        @alt = text.match(/--alt="([^"]*)"/i)[1]
      end
      if text =~ /--img-style="([^"]*)"/i
        @img_style = text.match(/--img-style="([^"]*)"/i)[1]
      end
      if text =~ /--class="([^"]*)"/i
        @class = text.match(/--class="([^"]*)"/i)[1]
      end
      if text =~ /--data="([^"]*)"/i
        @data = text.match(/--data="([^"]*)"/i)[1]
      end
      if text =~ /--thumb="([^"]*)"/i
        @thumb = text.match(/--thumb="([^"]*)"/i)[1]
      end


    end

    def render(context)
      url = context.registers[:page]["url"]
      relative = "../" * (url.split("/").length-1)
      src = File.join(relative, @path == nil ? '' : @path);
      thumbSrc = File.join(relative, @thumb == nil ? '' : @thumb);
      %{<a href="#{src}" data-lightbox="#{@data}" data-title="#{@title}"><img src="#{thumbSrc}" alt="#{@alt || @title}" class="#{@class}" style="#{@img_style}"/></a>}
    end
  end
end

Liquid::Template.register_tag('lightbox', Jekyll::LightboxTag)