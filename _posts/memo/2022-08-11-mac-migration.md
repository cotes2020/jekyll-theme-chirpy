---
categories:
  - memo
date: 2022-08-11 00:00:00 +0900
tags:
  - mac
  - vscode
  - zsh
title: macのセットアップ手順メモ
---

mac から mac へ移行した時の手順を記憶してる限り全部書いてみた。

## mac のシステム設定

- [カーソルの移動速度を速度を速くした](https://ushigyu.net/2015/05/07/how-to-faster-macbook-cursor/)
- [キーのリピート速度を早くした](https://support.apple.com/ja-jp/guide/mac-help/mchl0311bdb4/mac)
- [メニューバーに充電残量を表示した](https://support.apple.com/ja-jp/guide/mac-help/mchlp1115/mac)
- [キーボードの修飾キーで caps lock を esc にした](https://support.apple.com/ja-jp/guide/mac-help/mchlp1011/mac)
- spotlight 検索の対象をアプリのみに変更

## ブラウザ

- 元々使っていた Brave を同期させただけでブックマーク、拡張機能、パスワードを完全に移行できた。多分、Chrome にも似た機能はあると思う。

## shell の設定

- 先に brew, zinit, pyenv を入れておく
- 今まで使っていた [zshrc](https://github.com/yuji96/dotfiles/blob/master/.zshrc) をコピペして立ち上げるだけで再現できた。
- gitconfig ファイルもコピペで移行する。
- ssh key もコピペする。（本当は鍵作り直しの方が良いのかもしれないけど）

zinit でインストールできるおすすめカスタマイズ

- [`powerlevel10k`](https://github.com/romkatv/powerlevel10k) : かなりかっこよくなる
- [`zsh-users/zsh-completions`](https://github.com/zsh-users/zsh-completions) : git とかの有名コマンドの補完をしてくれる
- [`zsh-users/zsh-autosuggestions`](https://github.com/zsh-users/zsh-autosuggestions) : 過去に使用したコマンドを補完してくれる
- [`zdharma-continuum/fast-syntax-highlighting`](https://github.com/zdharma-continuum/fast-syntax-highlighting) : ちょっとかっこよくなる
- [`zsh-users/zsh-history-substring-search`](https://github.com/zsh-users/zsh-history-substring-search) : 過去に使用したコマンドの検索ができる

## vscode

- setting sync という標準機能で拡張機能、キーバインド、俺流 settings.json が一発で移行できた。感動。ただし、先に環境構築をしないと「python がありません」とかのエラーがいっぱい出てくるので注意。
- [`code` command を有効化する](https://qiita.com/naru0504/items/c2ed8869ffbf7682cf5c)

## 入力ソースを Google 日本語入力にする

mac の日本語入力は変換候補が使っているうちにバ ○ になっていくので、Google 日本語入力に変更するのがおすすめ。

- [公式 HP](https://www.google.co.jp/ime/)
- [インストール手順](https://shimautablog.com/mac_googleime_install/)
- [入力ソースからデフォルトの英語を消す方法](https://www.karakaram.com/deleting-alphanumeric-input-sources-on-macos-bigsur/)

僕が M1 mac を買った当初はアプリによって動かなかったのにひどく落胆したが、気づいたらアプデされて使えるようになっていた。Google 大好き。

## アプリをインストールする

- zoom, line, slack, vscode, brave, office, deepl, etc.

## メール設定

- メールアプリでログインするだけ

## データ移動

ここから mac 移行の話。

移行アシスタントは起動が遅く、全ファイル移そうとする。大抵の場合もういらないファイルが合ったりするので、新しい mac をハードディスクとして読み込んでコピったほうが良さげ
[https://support.apple.com/ja-jp/guide/mac-help/mchlb37e8ca7/mac](https://support.apple.com/ja-jp/guide/mac-help/mchlb37e8ca7/mac)
