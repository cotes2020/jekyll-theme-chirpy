FROM ubuntu:22.04

RUN apt update && \
    apt-get install -y ruby-full build-essential zlib1g-dev git libmariadb-dev

RUN useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo -u 1000 ubuntu
USER ubuntu
WORKDIR /home/ubuntu

ENV GEM_HOME="/home/ubuntu/gems"
ENV PATH="/home/ubuntu/gems/bin:$PATH"

RUN gem install jekyll bundler

RUN git clone  https://github.com/opslogic/jekyll-theme-chirpy.git -b tr-blog --single-branch

WORKDIR /home/ubuntu/jekyll-theme-chirpy

RUN bundle install

RUN rm -rf /home/ubuntu/jekyll-theme-chirpy
