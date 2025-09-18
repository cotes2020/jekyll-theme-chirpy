# frozen_string_literal: true

Gem::Specification.new do |spec|
  spec.name          = "jekyll-theme-mammut"
  spec.version       = "7.3.1"
  spec.authors       = ["Toby Wang"]
  spec.email         = ["tobytywang@163.com"]

  spec.summary       = "A minimal, responsive, and feature-rich Jekyll theme for technical writing."
  spec.homepage      = "https://github.com/Tobytywang/jekyll-theme-mammut"
  spec.license       = "MIT"

  spec.files         = `git ls-files -z`.split("\x0").select { |f|
    f.match(%r!^((_(includes|layouts|sass|(data\/(locales|origin)))|assets)\/|README|LICENSE)!i)
  }

  spec.metadata = {
    "bug_tracker_uri"   => "https://github.com/Tobytywang/jekyll-theme-mammut/issues",
    "documentation_uri" => "https://github.com/Tobytywang/jekyll-theme-mammut/#readme",
    "homepage_uri"      => "https://cotes2020.github.io/chirpy-demo",
    "source_code_uri"   => "https://github.com/Tobytywang/jekyll-theme-mammut",
    "wiki_uri"          => "https://github.com/Tobytywang/jekyll-theme-mammut/wiki",
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
