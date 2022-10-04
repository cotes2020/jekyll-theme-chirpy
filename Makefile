.PHONY: local
local:
	docker run --rm \
      --volume="$$PWD:/srv/jekyll" \
      --volume="$$PWD/vendor/bundle:/usr/local/bundle" \
      --platform linux/amd64 \
      -it \
      --publish 4000:4000 jekyll/jekyll:4.2.0 jekyll serve