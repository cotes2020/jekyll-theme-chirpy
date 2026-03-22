#!/usr/bin/env ruby
# frozen_string_literal: true

# _plugins/llms-txt.rb
# Generate /llms.txt and /llms-full.txt for Jekyll sites following https://llmstxt.org
#
# Zero new dependencies. Uses Jekyll's site payload (posts, pages, tabs, collections).

module Jekyll
  class LlmsTxtGenerator < Generator
    safe true
    priority :low

    def generate(site)
      return if site.config['llms_txt'] == false

      pages = collect_pages(site)
      site.pages << build_llms_txt(site, pages)
      site.pages << build_llms_full_txt(site, pages)
    end

    private

    def collect_pages(site)
      pages = []

      site.posts.docs.each do |post|
        next unless post.published?
        pages << {
          title: post.data['title'].to_s,
          url: post.url,
          description: (post.data['excerpt'] || post.data['description'] || '').to_s.strip
        }
      end

      site.pages.each do |page|
        next if page.data['exclude_from_search']
        next if page.url.end_with?('.xml', '.json', '.txt')
        next if page.data['search'] == false
        next if page.url.end_with?('/llms.txt', '/llms-full.txt')

        title = (page.data['title'] || File.basename(page.url, '.*')).to_s
        pages << {
          title: title,
          url: page.url,
          description: (page.data['excerpt'] || page.data['description'] || '').to_s.strip
        }
      end

      pages.uniq { |p| p[:url] }
    end

    def build_llms_txt(site, pages)
      site_url = site.config['url'].to_s.sub(%r{/$}, '')
      title = site.config['title'] || site.config['name'] || 'Jekyll Site'
      description = site.config['description'] || ''

      lines = []
      lines << "# #{title}"
      lines << "> #{description}" if description && !description.empty?
      lines << ""

      sections = site.config['llms_txt_sections'] || default_sections
      sections.each do |section|
        section_pages = pages.select { |p| section_matches?(p, section) }
        next if section_pages.empty?

        lines << "## #{section['title']}"
        section_pages.each do |p|
          desc = p[:description].empty? ? '' : ": #{truncate(p[:description], 120)}"
          lines << "- [#{p[:title]}](#{site_url}#{p[:url]})#{desc}"
        end
        lines << ""
      end

      make_page(site, '/llms.txt', lines.join("\n"))
    end

    def build_llms_full_txt(site, pages)
      site_url = site.config['url'].to_s.sub(%r{/$}, '')
      title = site.config['title'] || site.config['name'] || 'Jekyll Site'
      description = site.config['description'] || ''

      lines = []
      lines << "# #{title}"
      lines << "> #{description}" if description && !description.empty?
      lines << ""

      pages.each do |p|
        lines << "## #{p[:title]}"
        lines << "URL: #{site_url}#{p[:url]}"
        lines << ""
        lines << p[:description]
        lines << ""
        lines << "---"
        lines << ""
      end

      make_page(site, '/llms-full.txt', lines.join("\n"))
    end

    def make_page(site, url_path, content)
      page = PageWithoutAFile.new(site, site.source, '', 'llms-txt-placeholder')
      page.url = url_path
      page.content = content
      page.data['layout'] = nil
      page.data['exclude_from_search'] = true
      page
    end

    def default_sections
      [
        { 'title' => 'Documentation', 'pattern' => '/^(about|docs|guide|help|reference|tab|archives|categories|tags)/i' },
        { 'title' => 'Blog Posts', 'pattern' => '/^\\/\\d{4}/' }
      ]
    end

    def section_matches?(page, section)
      pattern = section['pattern'] || section['title']
      return true if pattern.nil?

      regex = pattern.start_with?('/') ? Regexp.new(pattern[1..-2]) : /#{Regexp.escape(pattern)}/i
      regex.match?(page[:url]) || regex.match?(page[:title])
    end

    def truncate(text, max)
      text.length > max ? "#{text[0..max - 3]}..." : text
    end
  end
end
