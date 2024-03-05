# -*- encoding: utf-8 -*-
# stub: mercenary 0.4.0 ruby lib

Gem::Specification.new do |s|
  s.name = "mercenary".freeze
  s.version = "0.4.0".freeze

  s.required_rubygems_version = Gem::Requirement.new(">= 0".freeze) if s.respond_to? :required_rubygems_version=
  s.require_paths = ["lib".freeze]
  s.authors = ["Tom Preston-Werner".freeze, "Parker Moore".freeze]
  s.date = "2020-01-18"
  s.description = "Lightweight and flexible library for writing command-line apps in Ruby.".freeze
  s.email = ["tom@mojombo.com".freeze, "parkrmoore@gmail.com".freeze]
  s.homepage = "https://github.com/jekyll/mercenary".freeze
  s.licenses = ["MIT".freeze]
  s.required_ruby_version = Gem::Requirement.new(">= 2.4.0".freeze)
  s.rubygems_version = "3.5.3".freeze
  s.summary = "Lightweight and flexible library for writing command-line apps in Ruby.".freeze

  s.installed_by_version = "3.5.3".freeze if s.respond_to? :installed_by_version

  s.specification_version = 4

  s.add_development_dependency(%q<bundler>.freeze, [">= 0".freeze])
  s.add_development_dependency(%q<rake>.freeze, [">= 0".freeze])
  s.add_development_dependency(%q<rspec>.freeze, ["~> 3.0".freeze])
  s.add_development_dependency(%q<rubocop-jekyll>.freeze, ["~> 0.10.0".freeze])
end
