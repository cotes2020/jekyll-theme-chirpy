FROM ruby:2.4-slim

WORKDIR /app

COPY Gemfile .

RUN apt-get -y update && apt-get -y install python3 python3-pip make gcc  g++ git

RUN pip3 install ruamel.yaml

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN gem install jekyll -v3.8

RUN bundle install

CMD ["/bin/sh"]

ENTRYPOINT ["./tools/run.sh", ""]
