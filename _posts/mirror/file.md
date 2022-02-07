---
categories:
  - note
date: 2021-02-07 00:00:00 +0900
tags:
  - python
title: Python でファイルを相対的に読み込む方法
---

こんにちは。今まで、C++, JavaScript, R などの言語に触れたことはありますが、結局 Python に返ってきてしまう Yuji です。

# やりたいこと

今回の目標は、以下のような階層構造があったときに、`reader.py` から `data.csv` を読み込むというとてもシンプルなことです。

```
~/workspace/tmp_dir 
 ├── data.csv
 └── reader.py
```

（簡単そうでしょ？本当かな？）

# 普通にファイル名を指定した場合

では、そのままファイル名を指定してみると、
```python
import pandas as pd

try:
    pd.read_csv("data.csv")
    print("成功")
except FileNotFoundError:
    print("失敗")
```
```
> python reader.py
成功
```
もちろん問題なく成功しました。しかし少し意地悪をすると、
```
> cd ..
> python tmp_dir/reader.py
失敗
```
ファイルが見つからないと言われてしまいました。

# 原因

コマンドを見ると明らかですが、`cd ..` で一つ上の階層に移動したことで Python を実行した階層から見たファイルの位置が `data.csv` から `tmp_dir/data.csv` に変わったことが原因です。なので、`reader.py` の4行目を
```python
    pd.read_csv("tmp_dir/data.csv")
```
と変更すれば（一応）解決します。

しかし、Python を実行する階層は人によって変わったり、他のファイルに `import` されたりするかもしれないので、その度にコードを書き換えると GitHub 上で戦争が起きたりローカルの差分が荒れたりします。絶対パスも人によって異なるので解決策とは言えないでしょう。

ちなみに、Jupyter では階層を移動するという概念がないので、ファイル名の直指定でうまくいきます。（Colab ではあるみたいだけど。）

# 解決策

`__file__` と `pathlib`（または、`os.path` ）を使いましょう。
Python のコードを読んだり書いているときに `__file__` という変数を見たことはあるでしょうか。試しに `reader.py` をこれだけにして実行してみましょう。

```python
print(__file__)
```
```
> python reader.py
reader.py
> cd ..
> python tmp_dir/reader.py
tmp_dir/reader.py
```
Python を実行した階層から見たファイル名が出力されました。
では、次はこのようにすると、
```python
from pathlib import Path
print(Path(__file__).resolve().parent)
```
```
> python reader.py
~/workspace/tmp_dir
> cd ..
> python tmp_dir/reader.py
~/workspace/tmp_dir
```
どこから実行しても同じ出力が得られました。
`resolve` は相対パスを絶対パスに変換するメソッドで、`parent` は親ディレクトリを返します。つまり今回は、`reader.py` がある `tmp_dir` の絶対パスを取得できています。

最後に、これらを使って今回の目標だった `data.csv` を読み込んでみましょう。
```python
from pathlib import Path
import pandas as pd

try:
    parent = Path(__file__).resolve().parent
    pd.read_csv(parent.joinpath("data.csv"))
    print("成功")
except FileNotFoundError:
    print("失敗")
```
```
> python reader.py
成功
> cd ..
> python tmp_dir/reader.py
成功
```

これでどこから実行してもファイルを読み込むことが出来ました！

このような小技は適用範囲が広いので恩恵は大きいです。
「なんか今のやり方スッキリしないなー」という小さな煩わしさを妥協しないようにしてると小技も少しずつ身についていくと思います。
