# frozen_string_literal: true

lib = File.expand_path('lib', __dir__)
$LOAD_PATH.unshift(lib) unless $LOAD_PATH.include?(lib)
require 'yell/version'

Gem::Specification.new do |spec|
  spec.name          = 'yell'
  spec.version       = Yell::VERSION
  spec.authors       = ['Rudolf Schmidt']
  spec.license       = 'MIT'

  spec.summary     = 'Yell - Your Extensible Logging Library'
  spec.description = "Yell - Your Extensible Logging Library. Define multiple adapters, various log level combinations or message formatting options like you've never done before"
  spec.homepage    = 'https://github.com/rudionrails/yell'

  spec.metadata['homepage_uri'] = spec.homepage
  spec.metadata['source_code_uri'] = 'https://github.com/rudionrails/yell'

  # Specify which files should be added to the gem when it is released.
  # The `git ls-files -z` loads the files in the RubyGem that have been added into git.
  spec.files = Dir.chdir(File.expand_path(__dir__)) do
    `git ls-files -z`.split("\x0").reject { |f| f.match(%r{^(test|spec|features)/}) }
  end
  spec.bindir        = 'exe'
  spec.executables   = spec.files.grep(%r{^exe/}) { |f| File.basename(f) }
  spec.require_paths = ['lib']
end
