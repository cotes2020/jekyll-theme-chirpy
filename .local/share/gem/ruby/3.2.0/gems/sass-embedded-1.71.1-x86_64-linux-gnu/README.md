# Embedded Sass Host for Ruby

[![build](https://github.com/sass-contrib/sass-embedded-host-ruby/actions/workflows/build.yml/badge.svg)](https://github.com/sass-contrib/sass-embedded-host-ruby/actions/workflows/build.yml)
[![gem](https://badge.fury.io/rb/sass-embedded.svg)](https://rubygems.org/gems/sass-embedded)

This is a Ruby library that implements the host side of the [Embedded Sass protocol](https://github.com/sass/sass/blob/HEAD/spec/embedded-protocol.md).

It exposes a Ruby API for Sass that's backed by a native [Dart Sass](https://sass-lang.com/dart-sass) executable.

## Install

``` sh
gem install sass-embedded
```

## Usage

The Ruby API provides two entrypoints for compiling Sass to CSS.

- `Sass.compile` takes a path to a Sass file and return the result of compiling that file to CSS.

``` ruby
require 'sass-embedded'

result = Sass.compile('style.scss')
puts result.css

compressed = Sass.compile('style.scss', style: :compressed)
puts compressed.css
```

- `Sass.compile_string` takes a string that represents the contents of a Sass file and return the result of compiling that file to CSS.

``` ruby
require 'sass-embedded'

result = Sass.compile_string('h1 { font-size: 40px; }')
puts result.css

compressed = Sass.compile_string('h1 { font-size: 40px; }', style: :compressed)
puts compressed.css
```

See [rubydoc.info/gems/sass-embedded/Sass](https://rubydoc.info/gems/sass-embedded/Sass) for full API documentation.

---

Disclaimer: this is not an official Google product.
