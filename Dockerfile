FROM ruby:3.1.1

RUN apt-get update -qq && apt-get install -y build-essential nodejs

WORKDIR /workspace
COPY . /workspace

RUN gem install bundler jekyll

RUN bundle install

EXPOSE 4000

CMD ["bundle", "exec", "jekyll", "serve", "--host", "0.0.0.0"]

