draft:
	bundle exec jekyll draft $(n)

post:
	bundle exec jekyll post $(n) --timestamp-format "%Y-%m-%d %H:%M:%S %z" 

dev:
	bash tools/run.sh