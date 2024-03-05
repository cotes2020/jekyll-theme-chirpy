# frozen_string_literal: true

module HTMLProofer
  # https://stackoverflow.com/a/8812293
  class XpathFunctions
    def case_sensitive_equals(node_set, str_to_match)
      node_set.find_all { |node| node.to_s.== str_to_match.to_s }
    end
  end
end
