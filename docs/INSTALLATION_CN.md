# 必需环境安装流程（简明）

## 📦 安装 Ruby + Jekyll + Bundler

安装 Ruby（含 DevKit）

访问 rubyinstaller.org
，下载针对 Windows 的 Ruby + Devkit 版本。

安装过程中，建议保持默认选项。在安装结束时，勾选“运行 ridk install”以安装开发工具链。 
Zhao Zhengyang
+1

安装后，打开新的命令提示符（CMD），执行以下命令验证是否安装成功：

ruby -v
gem -v


如果成功，你将看到 Ruby 与 gem 的版本号。 
Zhao Zhengyang
+1

安装 Jekyll 和 Bundler

在命令行中执行：

gem install jekyll bundler


安装完成后，检查 Jekyll 是否可用：

jekyll -v


如果输出版本号，则说明安装成功。 
Zhao Zhengyang
+1

📂 获取项目 & 安装依赖

克隆项目仓库

在你希望放置项目的文件夹（比如 D:\Projects\）打开命令行，然后执行：

git clone https://github.com/RookiePhaseChangeBoss/RookiePhaseChangeBoss.github.io.git
cd RookiePhaseChangeBoss.github.io


如果你使用 GUI Git 工具，也可以用它克隆。

安装 Gem / Node /依赖

进入项目根目录后，先执行：

bundle install


这会根据 Gemfile 安装 Ruby 相关依赖。

注意：本项目中可能还含有前端构建工具（因为看到有 package.json、rollup.config.js、purgecss.js 等文件） 
GitHub

如果你想让前端相关功能（CSS/JS 构建、主题资源编译等）正确生效，请确保你已安装 Node.js + npm/yarn。推荐安装最新稳定版 Node.js。

🚀 本地运行 / 预览网站

在项目根目录下，在命令行执行：

bundle exec jekyll serve

该命令将启动一个本地服务器，通常可以通过 http://127.0.0.1:4000/ 访问你的网站预览页面。 
腾讯云
+1

若一切顺利，你应当能看到基于 Chirpy 主题渲染后的网站首页／文档。

⚠️ 如果遇到问题，比如主题未生效、CSS/JS 加载异常等，可能需要先运行前端构建（例如根据 package.json 使用 npm install + npm run build／类似命令），具体取决于该项目 README/Wiki 中的说明。


gem install bundler --source https://rubygems.org/ --http-proxy http://127.0.0.1:7890
$env:http_proxy = "http://127.0.0.1:7890"
$env:https_proxy = "http://127.0.0.1:7890"
