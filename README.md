# Local testing

Open Docker

Use WSL terminal

#### Install dependencies *Install gemfile dependences to /tmp/vendor with bundler*

```bash
docker run -t -v $PWD:/app -v /tmp/vendor:/vendor -w /app -e BUNDLE_PATH=/vendor ruby bundle
```

### Preview website *http://localhost:4000* 
```bash
docker run -t -v $PWD:/app -v /tmp/vendor:/vendor -w /app -e BUNDLE_PATH=/vendor -p 4000:4000 ruby bundle exec jekyll serve --watch --drafts --force_polling -H 0.0.0.0
```