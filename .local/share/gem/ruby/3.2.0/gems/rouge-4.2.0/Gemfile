# frozen_string_literal: true

source 'http://rubygems.org'

gemspec

gem 'rake'

gem 'minitest', '>= 5.0'
gem 'minitest-power_assert'
gem 'power_assert', '~> 2.0'

gem 'rubocop', '~> 1.0', '<= 1.11'
gem 'rubocop-performance'

# don't try to install redcarpet under jruby
gem 'redcarpet', platforms: :ruby

# Profiling
gem 'memory_profiler', require: false

# Needed for a Rake task
gem 'git'
gem 'yard'

group :development do
  gem 'pry'

  # docs
  gem 'github-markup'

  # for visual tests
  gem 'sinatra'

  # Ruby 3 no longer ships with a web server
  gem 'puma' if RUBY_VERSION >= '3'
  gem 'shotgun'
end
