#!/usr/bin/env bash
set -e

# -------------------------------------------------------
# Jekyll Chirpy 部署脚本
# 本脚本执行步骤：
# 1. jekyll build 生成静态文件到 _site
# 2. 将 _site 推送到 gh-pages 分支
# -------------------------------------------------------

echo "🛠 1/4 清理旧构建..."
rm -rf _site

echo "🧱 2/4 构建 Jekyll 网站..."
bundle exec jekyll build

echo "🚚 3/4 准备部署到 gh-pages..."

cd _site
git init
git add .
git commit -m "🚀 Deploy update $(date +'%Y-%m-%d %H:%M:%S')"

# ⚠️ 这里替换成你的 GitHub 仓库地址
git remote add origin https://github.com/coder-cjl/coder-cjl.github.io.git

# 使用 gh-pages 分支
git branch -M gh-pages

echo "🌐 4/4 推送到 GitHub Pages..."
git push -f origin gh-pages

cd ..

echo "✨ 部署成功！访问地址：https://coder-cjl.github.io/"
