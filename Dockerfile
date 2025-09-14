# Stage 1: The Builder
# Use a full Ruby image to build the Jekyll site
FROM ruby:3.2.3-alpine AS builder

RUN apk add --no-cache --update build-base git
# Install dependencies
WORKDIR /app
COPY Gemfile Gemfile.lock ./
RUN bundle config set deployment 'true' && \
    bundle config set without 'development test' && \
    bundle install

# Build the site
COPY . .
RUN JEKYLL_ENV=production bundle exec jekyll b

# Stage 2: The Web Server
# Use a tiny web server image for the final output
FROM nginx:alpine

# Remove default nginx site
RUN rm -rf /usr/share/nginx/html/*

# Copy *only the built site* from the previous "builder" stage
COPY --from=builder /app/_site /usr/share/nginx/html

# nginx runs on port 80 by default
EXPOSE 80