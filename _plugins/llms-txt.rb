     1|#!/usr/bin/env ruby
     2|# frozen_string_literal: true
     3|
     4|# _plugins/llms-txt.rb
     5|# Generate /llms.txt and /llms-full.txt for Jekyll sites following https://llmstxt.org
     6|#
     7|# Zero new dependencies. Uses Jekyll's site payload (posts, pages, tabs, collections).
     8|
     9|module Jekyll
    10|  class LlmsTxtGenerator < Generator
    11|    safe true
    12|    priority :low
    13|
    14|    def generate(site)
    15|      return if site.config['llms_txt'] == false
    16|
    17|      pages = collect_pages(site)
    18|      site.pages << build_page(site, 'llms.txt', build_llms_txt_content(site, pages))
    19|      site.pages << build_page(site, 'llms-full.txt', build_llms_full_txt_content(site, pages))
    20|    end
    21|
    22|    private
    23|
    24|    def collect_pages(site)
    25|      pages = []
    26|
    27|      posts = site.posts.respond_to?(:docs) ? site.posts.docs : []
    28|      posts.each do |post|
    29|        next if post.respond_to?(:published?) && !post.published?
    30|        pages << {
    31|          title: post.data['title'].to_s,
    32|          url: post.url,
    33|          description: (post.data['excerpt'] || post.data['description'] || '').to_s.strip
    34|        }
    35|      end
    36|
    37|      site.pages.each do |page|
    38|        next if page.data['exclude_from_search']
    39|        next if page.url.end_with?('.xml', '.json', '.txt')
    40|        next if page.data['search'] == false
    41|        next if page.url.end_with?('/llms.txt', '/llms-full.txt')
    42|
    43|        title = (page.data['title'] || File.basename(page.url, '.*')).to_s
    44|        pages << {
    45|          title: title,
    46|          url: page.url,
    47|          description: (page.data['excerpt'] || page.data['description'] || '').to_s.strip
    48|        }
    49|      end
    50|
    51|      pages.uniq { |p| p[:url] }
    52|    end
    53|
    54|    def build_llms_txt_content(site, pages)
    55|      site_url = site.config['url'].to_s.sub(%r{/$}, '')
    56|      title = site.config['title'] || site.config['name'] || 'Jekyll Site'
    57|      description = site.config['description'] || ''
    58|
    59|      lines = []
    60|      lines << "# #{title}"
    61|      lines << "> #{description}" if description && !description.empty?
    62|      lines << ""
    63|
    64|      sections = site.config['llms_txt_sections'] || default_sections
    65|      sections.each do |section|
    66|        section_pages = pages.select { |p| section_matches?(p, section) }
    67|        next if section_pages.empty?
    68|
    69|        lines << "## #{section['title']}"
    70|        section_pages.each do |p|
    71|          desc = p[:description].empty? ? '' : ": #{truncate(p[:description], 120)}"
    72|          lines << "- [#{p[:title]}](#{site_url}#{p[:url]})#{desc}"
    73|        end
    74|        lines << ""
    75|      end
    76|
    77|      lines.join("\n")
    78|    end
    79|
    80|    def build_llms_full_txt_content(site, pages)
    81|      site_url = site.config['url'].to_s.sub(%r{/$}, '')
    82|      title = site.config['title'] || site.config['name'] || 'Jekyll Site'
    83|      description = site.config['description'] || ''
    84|
    85|      lines = []
    86|      lines << "# #{title}"
    87|      lines << "> #{description}" if description && !description.empty?
    88|      lines << ""
    89|
    90|      pages.each do |p|
    91|        lines << "## #{p[:title]}"
    92|        lines << "URL: #{site_url}#{p[:url]}"
    93|        lines << ""
    94|        lines << p[:description]
    95|        lines << ""
    96|        lines << "---"
    97|        lines << ""
    98|      end
    99|
   100|      lines.join("\n")
   101|    end
   102|
   103|    def build_page(site, filename, content)
   104|      page = PageWithoutAFile.new(site, site.source, '', filename)
   105|      page.content = content
   106|      page.data['layout'] = nil
   107|      page.data['exclude_from_search'] = true
   108|      page
   109|    end
   110|
   111|    def default_sections
   112|      [
   113|        { 'title' => 'Documentation', 'pattern' => '/^(about|docs|guide|help|reference|tab|archives|categories|tags)/i' },
   114|        { 'title' => 'Blog Posts', 'pattern' => '/^\\\\/\\\\d{4}/' }
   115|      ]
   116|    end
   117|
   118|    def section_matches?(page, section)
   119|      pattern = section['pattern'] || section['title']
   120|      return true if pattern.nil?
   121|
   122|      regex = pattern.start_with?('/') ? Regexp.new(pattern[1..-2]) : /#{Regexp.escape(pattern)}/i
   123|      regex.match?(page[:url]) || regex.match?(page[:title])
   124|    end
   125|
   126|    def truncate(text, max)
   127|      text.length > max ? "#{text[0..max - 3]}..." : text
   128|    end
   129|  end
   130|end
   131|