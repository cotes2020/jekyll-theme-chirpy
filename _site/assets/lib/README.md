# Static Assets for Chirpy Jekyll Theme

## Introduction

Static assets (libraries/plugins/web-fonts) required by the [_Chirpy_][chirpy] based website to run. It provides the opportunity to choose self-host assets in production or development mode.

## Usage

- If you want to use these assets only in local development:

  Go to the root of your site and clone the assets as follows.

  ```console
  $ cd assets/lib
  $ git submodule init
  $ git submodule update
  ```

  And then set your site configuration options:

  ```yml
  # _config.yml
  assets:
    self_host:
      enabled: true
      env: development
  ```

- If you expect the assets to be self-hosted when your website is published:

  Keep the `_config.yml` options as follows:

  ```yml
  # _config.yml
  assets:
    self_host:
      enabled: true
  ```

  And then update the GitHub Actions workflow in `.github/workflows/pages-deploy.yml`:

  ```diff
  steps:
    - name: Checkout
      uses: actions/checkout@v2
      with:
  +     submodules: true
  ```

[assets]: https://github.com/cotes2020/chirpy-static-assets
[chirpy]: https://github.com/cotes2020/jekyll-theme-chirpy
