---
title: github 블로그와 jekyll
date: 2023-08-20
categories: [blog]
tags: [jekyll, ruby]
---

```zsh
Unfortunately, an unexpected error occurred, and Bundler cannot continue.

First, try this link to see if there are any existing issue reports for this error:
https://github.com/rubygems/rubygems/search?q=You+don%27t+have+write+permissions+for+the+%2FLibrary%2FRuby%2FGems%2F2.6.0+directory.&type=Issues

```

```zsh
Your RubyGems version (3.0.3) has a bug that prevents `required_ruby_version` from working for Bundler. Any scripts that use `gem install bundler` will break as soon as Bundler drops support for your Ruby version. Please upgrade RubyGems to avoid future breakage and silence this warning by running `gem update --system 3.2.3`
bundler: command not found: jekyll
Install missing gem executables with `bundle install`
claire@Claireui-iMac YubinShin.github.io % bundle
```

```zsh
Gem::FilePermissionError: You don't have write permissions for the /Library/Ruby/Gems/2.6.0 directory.
```

```zsh
chruby: unknown Ruby: ruby-3.1.3
claire@Claireui-iMac YubinShin.github.io % jekyll serve
/Users/claire/.gem/ruby/3.2.0/gems/bundler-2.4.19/lib/bundler/runtime.rb:304:in `check_for_activated_spec!': You have already activated rexml 3.2.5, but your Gemfile requires rexml 3.2.6. Prepending `bundle exec` to your command may solve this. (Gem::LoadError)
        from /Users/claire/.gem/ruby/3.2.0/gems/bundler-2.4.19/lib/bundler/runtime.rb:25:in `block in setup'
        from /Users/claire/.gem/ruby/3.2.0/gems/bundler-2.4.19/lib/bundler/spec_set.rb:165:in `each'
        from /Users/claire/.gem/ruby/3.2.0/gems/bundler-2.4.19/lib/bundler/spec_set.rb:165:in `each'
        from /Users/claire/.gem/ruby/3.2.0/gems/bundler-2.4.19/lib/bundler/runtime.rb:24:in `map'
        from /Users/claire/.gem/ruby/3.2.0/gems/bundler-2.4.19/lib/bundler/runtime.rb:24:in `setup'
        from /Users/claire/.gem/ruby/3.2.0/gems/bundler-2.4.19/lib/bundler.rb:162:in `setup'
        from /Users/claire/.gem/ruby/3.2.0/gems/jekyll-4.3.2/lib/jekyll/plugin_manager.rb:52:in `require_from_bundler'
        from /Users/claire/.gem/ruby/3.2.0/gems/jekyll-4.3.2/exe/jekyll:11:in `<top (required)>'
        from /Users/claire/.rbenv/versions/3.2.2/bin/jekyll:25:in `load'
        from /Users/claire/.rbenv/versions/3.2.2/bin/jekyll:25:in `<main>'
claire@Claireui-iMac YubinShin.github.io % bundle exec
bundler: exec needs a command to run
claire@Claireui-iMac YubinShin.github.io % bundle exec jekyll serve
```

https://www.irgroup.org/posts/jekyll-chirpy/

https://jekyllrb.com/docs/installation/macos/

https://frhyme.github.io/others/jekyll_serve_not_work/

https://computer-science-student.tistory.com/388

```zsh

claire@Claireui-iMac YubinShin.github.io % jekyll serve
/Users/claire/.gem/ruby/3.2.0/gems/bundler-2.4.19/lib/bundler/runtime.rb:304:in `check_for_activated_spec!': You have already activated rexml 3.2.5, but your Gemfile requires rexml 3.2.6. Prepending `bundle exec` to your command may solve this. (Gem::LoadError)
        from /Users/claire/.gem/ruby/3.2.0/gems/bundler-2.4.19/lib/bundler/runtime.rb:25:in `block in setup'
        from /Users/claire/.gem/ruby/3.2.0/gems/bundler-2.4.19/lib/bundler/spec_set.rb:165:in `each'
        from /Users/claire/.gem/ruby/3.2.0/gems/bundler-2.4.19/lib/bundler/spec_set.rb:165:in `each'
        from /Users/claire/.gem/ruby/3.2.0/gems/bundler-2.4.19/lib/bundler/runtime.rb:24:in `map'
        from /Users/claire/.gem/ruby/3.2.0/gems/bundler-2.4.19/lib/bundler/runtime.rb:24:in `setup'
        from /Users/claire/.gem/ruby/3.2.0/gems/bundler-2.4.19/lib/bundler.rb:162:in `setup'
        from /Users/claire/.gem/ruby/3.2.0/gems/jekyll-4.3.2/lib/jekyll/plugin_manager.rb:52:in `require_from_bundler'
        from /Users/claire/.gem/ruby/3.2.0/gems/jekyll-4.3.2/exe/jekyll:11:in `<top (required)>'
        from /Users/claire/.rbenv/versions/3.2.2/bin/jekyll:25:in `load'
        from /Users/claire/.rbenv/versions/3.2.2/bin/jekyll:25:in `<main>'


claire@Claireui-iMac YubinShin.github.io % bundle exec
bundler: exec needs a command to run


claire@Claireui-iMac YubinShin.github.io % bundle exec jekyll serve

```

## VScode Extension

https://marketplace.visualstudio.com/items?itemName=yzhang.markdown-all-in-one