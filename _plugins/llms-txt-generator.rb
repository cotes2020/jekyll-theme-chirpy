#!/usr/bin/env ruby
# frozen_string_literal: true
 
#
# Generate /llms.txt and optionally /llms-full.txt for LLM-friendly site content.
# Spec: https://llmstxt.org/
#
# Contacts are auto-read from _data/contact.yml (no config needed).
#
# ── _config.yml ──────────────────────────────────────────────────────
#
#    llms_txt:
#      # Required
#      enabled: false                     # false or omitted = plugin does nothing
#
#      # Optional — omit or leave empty to use defaults.
#      # llms_txt `enabled` must be 'true' for these to apply.
#      full: true                        # generate llms-full.txt (default: false)
#      title: "My Site"                  # default: site.title → "Site"
#      description: "About my site"      # default: site.description → ""
#
#      # Optional — extra links shown under "## Optional" section.
#      # Per the llmstxt.org spec, LLMs may skip this section when context is limited.
#      # Omit or leave empty to skip this section.
#      # optional:
#      #   - name: RSS Feed
#      #     url: https://yourblog.com/feed.xml
#      #     description: Subscribe to new posts
#      #   - name: GitHub
#      #     url: https://github.com/yourname
#      #     description: Open source projects and code
#      #   - name: Resume
#      #     url: https://yourblog.com/resume.pdf
#      #     description: Professional background and experience
#
# ─────────────────────────────────────────────────────────────────────
#
# Minimal config (just these two lines):
#
#   llms_txt:
#     enabled: true
#

require "nokogiri"

module Jekyll
  class LlmsTxtGenerator < Generator
    safe true
    priority :low

    def generate(site)
      return unless enabled?(site)

      config = site.config["llms_txt"] || {}
      posts = site.posts.docs.sort_by(&:date).reverse
      pages = site.collections["tabs"]&.docs&.sort_by { |t| t.data["order"] || 0 } || []
      contacts = resolve_contacts(site)

      site.pages << build_llms_txt(site, config, posts, pages, contacts)
      site.pages << build_llms_full_txt(site, config, posts, pages, contacts) if config["full"]
    end

    private

    def enabled?(site)
      site.config.dig("llms_txt", "enabled") == true
    end

    def site_url(site)
      url = site.config["url"].to_s.chomp("/")
      base = site.config["baseurl"].to_s.chomp("/")
      "#{url}#{base}"
    end

    # ── Contact resolution ──────────────────────────────────────────────

    # Reads _data/contact.yml and resolves each entry to a name + URL pair.
    # Entries without a resolvable URL are silently skipped.
    def resolve_contacts(site)
      raw = site.data["contact"]
      return [] if raw.nil? || !raw.is_a?(Array)

      raw.each_with_object([]) do |entry, list|
        type = entry["type"].to_s
        url = entry["url"].to_s.strip
        url = infer_contact_url(site, type) if url.empty?
        next if url.empty?

        url = "#{site_url(site)}#{url}" if url.start_with?("/")
        list << { "name" => format_contact_name(type), "url" => url }
      end
    end

    # Tries to build a URL from _config.yml values when contact.yml
    # doesn't have an explicit url field.
    def infer_contact_url(site, type)
      case type
      when "github"
        username = site.config.dig("github", "username").to_s.strip
        username.empty? ? find_social_link(site, "github.com") : "https://github.com/#{username}"
      when "twitter"
        username = site.config.dig("twitter", "username").to_s.strip
        username.empty? ? find_social_link(site, "twitter.com") : "https://twitter.com/#{username}"
      when "email"
        email = site.config.dig("social", "email").to_s.strip
        email.empty? ? "" : "mailto:#{email}"
      when "rss"
        "#{site_url(site)}/feed.xml"
      else
        find_social_link(site, type)
      end
    end

    # Scans social.links array in _config.yml for a URL containing the keyword.
    def find_social_link(site, keyword)
      links = site.config.dig("social", "links")
      return "" unless links.is_a?(Array)

      links.find { |l| l.to_s.include?(keyword) }.to_s
    end

    # Turns a contact type slug into a readable name.
    def format_contact_name(type)
      type.split(/[-_]/).map(&:capitalize).join(" ")
    end

    # ── Section builders ────────────────────────────────────────────────

    def build_llms_txt(site, config, posts, pages, contacts)
      content = String.new
      content << header(site, config)
      content << pages_section(site, pages) unless pages.empty?
      content << posts_section(site, posts) unless posts.empty?
      content << contacts_section(contacts) unless contacts.empty?
      content << optional_section(config)

      make_page(site, "llms.txt", content)
    end

    def build_llms_full_txt(site, config, posts, pages, contacts)
      content = String.new
      content << header(site, config)

      base = site_url(site)

      unless pages.empty?
        content << "## Pages\n\n"
        pages.each do |page|
          title = page.data["title"] || page.basename
          url = "#{base}#{page.url}"
          content << "### [#{title}](#{url})\n\n"
          content << strip_html(page.content).strip << "\n\n"
        end
      end

      unless posts.empty?
        content << "## Posts\n\n"
        posts.each do |post|
          title = post.data["title"] || post.basename_without_ext
          url = "#{base}#{post.url}"
          desc = post.data["description"].to_s.strip
          content << "### [#{title}](#{url})\n\n"
          content << "> #{desc}\n\n" unless desc.empty?
          content << strip_html(post.content).strip << "\n\n"
        end
      end

      content << contacts_section(contacts) unless contacts.empty?
      content << optional_section(config)

      make_page(site, "llms-full.txt", content)
    end

    def header(site, config)
      title = config["title"] || site.config["title"] || "Site"
      desc = config["description"] || site.config["description"] || ""

      out = String.new("# #{title}\n\n")
      out << "> #{desc}\n\n" unless desc.empty?
      out
    end

    def pages_section(site, pages)
      base = site_url(site)
      out = String.new("## Pages\n\n")
      pages.each do |page|
        title = page.data["title"] || page.basename
        url = "#{base}#{page.url}"
        out << "- [#{title}](#{url})\n"
      end
      out << "\n"
    end

    def posts_section(site, posts)
      base = site_url(site)
      out = String.new("## Posts\n\n")
      posts.each do |post|
        title = post.data["title"] || post.basename_without_ext
        url = "#{base}#{post.url}"
        desc = post.data["description"].to_s.strip
        out << "- [#{title}](#{url})"
        out << ": #{desc}" unless desc.empty?
        out << "\n"
      end
      out << "\n"
    end

    def contacts_section(contacts)
      out = String.new("## Contacts\n\n")
      contacts.each do |c|
        out << "- [#{c['name']}](#{c['url']})\n"
      end
      out << "\n"
    end

    def optional_section(config)
      extras = config["optional"] || []
      return "" if extras.empty?

      out = String.new("## Optional\n\n")
      extras.each do |item|
        out << "- [#{item['name']}](#{item['url']})"
        out << ": #{item['description']}" if item["description"]
        out << "\n"
      end
      out << "\n"
    end

    def strip_html(text)
      Nokogiri::HTML.fragment(text).text
    end

    def make_page(site, name, content)
      page = PageWithoutAFile.new(site, site.source, "", name)
      page.data = {
        "layout" => nil,
        "sitemap" => false,
        "permalink" => "/#{name}"
      }
      page.content = content
      page.output = content
      page
    end
  end
end
