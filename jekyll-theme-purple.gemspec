# frozen_string_literal: true

Gem::Specification.new do |spec|
  spec.name          = "jekyll-theme-purple"
  spec.version       = "1.0.0"
  spec.authors       = ["Cotes Chung", "Vishwa R"]
  spec.email         = ["cotes.chung@gmail.com", "vishwajenvish@gmail.com"]

  spec.summary       = "A minimal, responsive, and feature-rich Jekyll theme for technical writing."
  spec.homepage      = "https://github.com/kyroceus/jekyll-theme-purple"
  spec.license       = "MIT"

  spec.files         = `git ls-files -z`.split("\x0").select { |f|
    f.match(%r!^((_(includes|layouts|sass|(data\/(locales|origin)))|assets)\/|README|LICENSE)!i)
  }

  spec.metadata = {
    "bug_tracker_uri"   => "https://github.com/kyroceus/jekyll-theme-purple/issues",
    "documentation_uri" => "https://github.com/kyroceus/jekyll-theme-purple/#readme",
    "homepage_uri"      => "https://github.com/kyroceus/jekyll-theme-purple",
    "source_code_uri"   => "https://github.com/kyroceus/jekyll-theme-purple",
    "wiki_uri"          => "https://github.com/kyroceus/jekyll-theme-purple/wiki",
    "plugin_type"       => "theme"
  }

  spec.required_ruby_version = "~> 3.1"

  spec.add_runtime_dependency "jekyll", "~> 4.3"
  spec.add_runtime_dependency "jekyll-paginate", "~> 1.1"
  spec.add_runtime_dependency "jekyll-seo-tag", "~> 2.8"
  spec.add_runtime_dependency "jekyll-archives", "~> 2.2"
  spec.add_runtime_dependency "jekyll-sitemap", "~> 1.4"
  spec.add_runtime_dependency "jekyll-include-cache", "~> 0.2"

end
