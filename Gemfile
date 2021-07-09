# frozen_string_literal: true

source "https://rubygems.org"

gem "jekyll", "~> 4.1"
gem "jekyll-theme-chirpy"

# Windows and JRuby does not include zoneinfo files, so bundle the tzinfo-data gem
# and associated library.

install_if -> { RUBY_PLATFORM =~ %r!mingw|mswin|java! } do
  gem "tzinfo", "~> 1.2"
  gem "tzinfo-data"
end

group :jekyll_plugins do
  gem "jekyll-paginate"
  end

# Performance-booster for watching directories on Windows
gem  "wdm", "~> 0.1.1", :install_if => Gem.win_platform?
gem "webrick", "~> 1.7"
