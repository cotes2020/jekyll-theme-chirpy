#!/usr/bin/env ruby
#
# Auto-generate tags for headers in grammar section

module Jekyll
    class HeaderTagGenerator < Generator
      priority :low
  
      def generate(site)
        site.posts.docs.each do |post|
          next unless post.content
  
          # Extract headers (e.g., #### 1. **～せい**)
          headers = post.content.scan(/^####\s+\d+\.\s+\*\*(.*?)\*\*/).flatten
  
          # Assign extracted headers as tags
          post.data['tags'] ||= []
          headers.each do |header|
            tag = header.strip
            post.data['tags'] << tag unless post.data['tags'].include?(tag)
          end
  
          # Ensure tags are unique
          post.data['tags'].uniq!

        end
      end
    end
  
    class HeaderLinkTagGenerator < Liquid::Tag
      def initialize(tag_name, text, tokens)
        super
        @text = text.strip
      end
  
      def render(context)
        site = context.registers[:site]
        posts = site.posts.docs.select { |post| post.data['tags'].include?(@text) }
  
        links = posts.map do |post|
          # Create a link to the header within the post
          post_url = context.registers[:site].config['baseurl'] + post.url
          "<a href='#{post_url}##{@text}'>#{@text}</a>"
        end
  
        links.join(', ')
      end
    end
  end
  
  Liquid::Template.register_tag('header_link', Jekyll::HeaderLinkTagGenerator)
  
