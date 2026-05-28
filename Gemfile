source "https://rubygems.org"

gem "jekyll", "~> 4.3"
# Pin to the v2 line — v3 pulls sass-embedded which needs a C toolchain.
# v2 uses sassc and ships precompiled binaries.
gem "jekyll-sass-converter", "~> 2.2"

group :jekyll_plugins do
  gem "jekyll-seo-tag", "~> 2.8"
  gem "jekyll-sitemap", "~> 1.4"
end

# Windows / JRuby tzinfo support
platforms :mingw, :x64_mingw, :mswin, :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
end
