#!/usr/bin/env ruby
#
# Auto-generate tags for headers in grammar section

module Jekyll
    class HeaderTagGenerator < Generator
      priority :high
  
      def generate(site)
        site.posts.docs.each do |post|
          next unless post.content
  
          # Extract headers (e.g., #### 1. **～せい**)
          headers = post.content.scan(/^####[\d\s\.]*\*\*(.*?)\*\*.*$/).flatten
  
          # Assign extracted headers as tags
          post.data['tags'] ||= []
          headers.each do |header|
            tag = header.strip
            post.data['tags'] << tag unless post.data['tags'].include?(tag)
          end
  
          # Ensure tags are unique
          post.data['tags'].uniq!
        end

        all_tags = site.posts.docs.flat_map { |post| post.data['tags'] }.uniq
        all_tags.each do |tag|
          site.pages << TagPage.new(site, tag)
        end
      end
    end
  
    class TagPage < Jekyll::Page
      def initialize(site, tag)
        @site = site
        @base = site.source
        @dir  = 'tags'
        @name = "#{tag}.md"
    
        self.process(@name)
        
        # Initialize data hash with default front matter
        self.data ||= {}
        self.data.merge!(
          'layout' => 'tag',
          'title' => tag,
          'tag' => tag,
          'permalink' => "/tags/#{tag}/"
        )
    
        # Set proper output extension
        self.ext = '.html'
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
  
