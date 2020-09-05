FROM ruby:alpine3.11

ARG WORKDIR=/app
ARG APP_PORT=4000
ENV APP_PORT=${APP_PORT}
ARG APP_TIMEZONE=Asia/Shanghai

RUN ln -snf /usr/share/zoneinfo/${APP_TIMEZONE} /etc/localtime \
	&& echo ${APP_TIMEZONE} > /etc/timezone

RUN apk add --no-cache make tzdata gcc g++ coreutils bash \
	file git autoconf automake libtool gettext gettext-dev texinfo curl

WORKDIR ${WORKDIR}

COPY Gemfile ${WORKDIR}

RUN bundle install

EXPOSE ${APP_PORT}
ENTRYPOINT [".docker/entrypoint.sh"]
CMD ["serve"]
