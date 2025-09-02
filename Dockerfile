FROM ruby:latest AS ruby-base

WORKDIR /app

RUN apt-get install git && gem install bundler

COPY Gemfile *.gemspec ./

RUN bundle install && git config --global --add safe.directory /app

COPY . .

FROM node:latest AS node-base

WORKDIR /app

COPY package.json ./

RUN npm install

COPY . .

EXPOSE 4000 35729
