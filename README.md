# jwsung91.github.io

[jwsung91.github.io](https://jwsung91.github.io)

This is a technical blog focused on software engineering topics. The blog is built using the [Chirpy](https://github.com/cotes2020/jekyll-theme-chirpy) theme for [Jekyll](https://jekyllrb.com).

## Using Docker container

### Build 

```zsh
docker build --no-cache -t my_space .
```

### Run

```zsh
docker run -it --rm -p 4000:4000 -v $(pwd):/workspace my_space
```