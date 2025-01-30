#!/usr/bin/env ruby
#
# Auto-generate tags for headers in grammar section

module Jekyll
    class HeaderTagGenerator < Generator
      priority :high

      def generate(site)
        site.posts.docs.each do |post|
          next unless post.content

          post.data['tags'] ||= []
          post.data['header_slugs'] ||= {}

          post.content.each_line do |line|
            next unless line.start_with?('#### ')

            header_line = line.strip.sub(/^#### /, '')
            tag_match = header_line.match(/\*\*(.*?)\*\*/)
            next unless tag_match

            # Extract the tag
            tag = tag_match[1]

            # Generate slug from full header text
            full_header_text = header_line.gsub(/\*\*/, '').strip
            slug = Jekyll::Utils.slugify(full_header_text)

            # Store slugified tag and slug
            # 
            # for unknown reasons, posts always have header slugs as nil
            post.data['tags'] << tag unless post.data['tags'].include?(tag)
            post.data['header_slugs'][tag] = slug
          end

          post.data['tags'].uniq!
        end

        # Create tag pages with slugified tags
        all_tags = site.posts.docs.flat_map { |post| post.data['tags'] }.uniq
        all_tags.each do |raw_tag|
          tag = Jekyll::Utils.slugify(raw_tag) # Slugify for filename safety
          site.pages << TagPage.new(site, tag, raw_tag) # Pass both versions
        end
      end
    end

    class TagPage < Jekyll::Page
      def initialize(site, slugified_tag, display_tag)
        @site = site
        @base = site.source
        @dir  = 'tags'
        @name = "#{slugified_tag}.md" # Use slugified name for filename
        
        self.process(@name)
        
        self.data ||= {}
        self.data.merge!(
          'layout' => 'tag',
          'title' => display_tag, # Original display name
          'tag' => slugified_tag, # Slugified version for lookup
          'permalink' => "/tags/{slugified_tag}/"
        )
        
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
  
