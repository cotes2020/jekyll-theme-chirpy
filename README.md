# Homepage for Matrix Org

## How to add content

+ Add new project: `_tabs/projects.md`
+ Edit About page: `_tabs/about.md`
+ Add new post: `_posts/`
+ Add new project in the home page: `index.md`
+ Edit other contents in the home page: `_layouts/home.html`
+ Most other text: `_config.yaml`
+ Add new images: `assets/img/`
+ Website icon: `assets/img/favicons/`

## How to deploy locally
### Install Ruby>=3.0.0 & Node.js
#### macOS
```bash
brew install ruby node
brew link --overwrite ruby
```
then open a new terminal session.
#### Ubuntu
```bash
sudo apt-get install ruby-full build-essential zlib1g-dev nodejs npm

echo '# Install Ruby Gems to ~/gems' >> ~/.bashrc
echo 'export GEM_HOME="$HOME/gems"' >> ~/.bashrc
echo 'export PATH="$HOME/gems/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Install Dependencies
```bash
npm install
npm run build
gem install jekyll bundler
bundle install
```
### Run locally
```bash
bundle exec jekyll serve [--host 0.0.0.0]
```

[//]: # (## Credits)

[//]: # ()
[//]: # (### Contributors)

[//]: # ()
[//]: # (Thanks to [all the contributors][contributors] involved in the development of the project!)

## License

This project is published under [MIT License][license].
