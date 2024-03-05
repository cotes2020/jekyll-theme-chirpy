require 'logger'

class Logger

  def level_with_yell=( level )
    self.level_without_yell= level.is_a?(Yell::Level) ? Integer(level) : level
  end
  alias_method :level_without_yell=, :level=
  alias_method :level=, :level_with_yell=

  def add_with_yell( severity, message = nil, progname = nil, &block )
    add_without_yell(Integer(severity), message, progname, &block)
  end
  alias_method :add_without_yell, :add
  alias_method :add, :add_with_yell

end
