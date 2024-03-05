require 'bundler/gem_tasks'

# Run stuff in the examples folder
desc "Run examples"
task :examples do
  require 'benchmark'

  seconds = Benchmark.realtime do
    Dir[ './examples/*.rb' ].each { |file| puts "\n\n=== Running #{file} ==="; require file }
  end

  puts "\n\t[ Examples took #{seconds} seconds to run ]"
end

