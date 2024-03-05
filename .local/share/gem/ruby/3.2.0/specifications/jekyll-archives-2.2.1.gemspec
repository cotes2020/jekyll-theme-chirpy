# -*- encoding: utf-8 -*-
# stub: jekyll-archives 2.2.1 ruby lib

Gem::Specification.new do |s|
  s.name = "jekyll-archives".freeze
  s.version = "2.2.1".freeze

  s.required_rubygems_version = Gem::Requirement.new(">= 0".freeze) if s.respond_to? :required_rubygems_version=
  s.require_paths = ["lib".freeze]
  s.authors = ["Alfred Xing".freeze]
  s.date = "2019-03-23"
  s.description = "Automatically generate post archives by dates, tags, and categories.".freeze
  s.homepage = "https://github.com/jekyll/jekyll-archives".freeze
  s.licenses = ["MIT".freeze]
  s.required_ruby_version = Gem::Requirement.new(">= 2.3.0".freeze)
  s.rubygems_version = "3.5.3".freeze
  s.summary = "Post archives for Jekyll.".freeze

  s.installed_by_version = "3.5.3".freeze if s.respond_to? :installed_by_version

  s.specification_version = 4

  s.add_runtime_dependency(%q<jekyll>.freeze, [">= 3.6".freeze, "< 5.0".freeze])
  s.add_development_dependency(%q<bundler>.freeze, [">= 0".freeze])
  s.add_development_dependency(%q<minitest>.freeze, [">= 0".freeze])
  s.add_development_dependency(%q<rake>.freeze, [">= 0".freeze])
  s.add_development_dependency(%q<rdoc>.freeze, [">= 0".freeze])
  s.add_development_dependency(%q<rubocop-jekyll>.freeze, ["~> 0.9".freeze])
  s.add_development_dependency(%q<shoulda>.freeze, [">= 0".freeze])
end
