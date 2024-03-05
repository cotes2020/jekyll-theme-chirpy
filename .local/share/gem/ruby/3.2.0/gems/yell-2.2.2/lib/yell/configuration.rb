require 'erb'
require 'yaml'

module Yell #:nodoc:

  # The Configuration can be used to setup Yell before
  # initializing an instance.
  class Configuration

    def self.load!( file )
      yaml = YAML.load( ERB.new(File.read(file)).result )

      # in case we have ActiveSupport
      if defined?(ActiveSupport::HashWithIndifferentAccess)
        yaml = ActiveSupport::HashWithIndifferentAccess.new(yaml)
      end

      yaml[Yell.env] || {}
    end

  end
end

