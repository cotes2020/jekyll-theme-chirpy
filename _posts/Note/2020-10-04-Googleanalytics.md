---
title: Implement Google Analytics to blog
date: 2020-10-04 11:11:11 -0400
# description: IT Blog Pool
categories: [Note]
# img: /assets/img/sample/rabbit.png
tags:
---


[toc]


---

# Implement Google Analytics to blog

```
Project ID
myo****site-291718
```

---

## Google APIs 创建项目

用 Google 账户登陆 **Google APIs Dashboard**
1. <kbd>Create Project</kbd> 新建一个 Project
   1. 起名: cotes-blog-ga，
   2. **Location**: 默认为 No organization。
2. 新建完毕后，为项目开启 API 和服务。<kbd>+ ENABLE APIS AND SERVICES</kbd> 进入API Library
3. 搜索栏中搜关键词 `analytic` 即可找到`Analytics API`，点击 `Enable`
4. 开启 API 后页面会自动回到 Dashboard，根据 ⚠️ 信息提示点击 <kbd>Create credentials</kbd> 为 API 创建 credentials。
5. 创建页面作如下操作：
   1. find out what kind of credentials needed:
      1. Which API are you using? `Google Analytics API`
      2. Where will you be calling the API from? `Web browser(Javascript)`
      3. What data will you be accessing? `User data`
   2. Create an OAuth 2.0 client ID
      1. Client ID 自定义命名: `blog-oauth`
      2. Restrictions 两项暂时留空，往后将会写入 GAE 的项目地址。
   3. Set up the OAuth 2.0 consent screen
      1. Email 保持默认值
      2. 产品名称自定义命名，不与其他公司产品重名即可，例笔者为 cotes-blog-ga
   4. Download credentials
      1. 视个人需要决定下载与否，
      2. `Client ID	318175415936-rdlkiaaf422e7kuenfq3blrnv0s5rn64.apps.googleusercontent.com`
      3. 供 SuperProxy 使用的 Client ID，Client secret 都可以在 Dashboard 直接查看。
   5. `完成后即可生成新 OAuth 2.0 client ID`:

---

## 下载配置 SuperProxy

**安装 Python 27**

---

**安装 Cloud SDK for Python**

```bash
$ python -V
Python 2.7.16
# Cloud SDK requires Python. Supported versions are 3.5 to 3.7, and 2.7.9 or higher.

# download the install filecd
./google-cloud-sdk/install.sh
./google-cloud-sdk/bin/gcloud init

export PATH="/Users/luo/google-cloud-sdk/bin:$PATH"
```

**下载 SuperProxy 项目**

[github](https://github.com/googleanalytics/google-analytics-super-proxy)

1. 修改 <kbd>src/config.py</kbd>：
   1. `OAUTH_CLIENT_ID` 与 `OAUTH_CLIENT_SECRET`，填入创建的 Client ID 与 Client secret。
   2. `PROJECT_ID` 在 Google APIs Dashboard 或者其他任一 GCP 页面中，点击顶栏项目名即可查看。
   3. `OAUTH_REDIRECT_URI` 填入 GAE 派发的免费域名
      1. 默认在地址尾部添加了 `/admin/auth`，
      2. 所以 URI 全貌为：`https://PROJECT_ID.appspot.com/admin/auth`。
   4. 返回上一步的 Credentials，点击 `OAuth 2.0 client IDs` 中的 `OAuth ID`，在设置页面的 `Authorized redirect URIs` 填入 `SuperProxy` 中 `OAUTH_REDIRECT_URI` 的完整地址，例如：`https://cotes-blog-ga-214617.appspot.com/admin/auth`。

2. 修改 <kbd>src/app.yaml</kbd>:
   1. 首部两行：application与version，在 Cloud SDK 213.0.0 中已经标记为无效字段了，需要将其删除，否则部署时会出现警告而导致中断。


---

**上传 SuperProxy 至 GAE**

```bash
enable CloudBuild API
enable Cloud Datastore API
# need billing account
# https://console.developers.google.com/apis/api/datastore.googleapis.com/overview?project=myochosite-291718

gcloud app deploy app.yaml index.yaml --project myochosite-291718
# chose location
#  [14] us-central
#  [15] us-east1
#  [16] us-east4
#  [17] us-west2
#  [18] us-west3
#  [19] us-west4
#  [20] cancel
# Please enter your numeric choice:  15
# Updating config [index]...done.
# Indexes are being rebuilt. This may take a moment.
# You can stream logs from the command line by running:
#   $ gcloud app logs tail -s default
# To view your application in the web browser run:
#   $ gcloud app browse

```

---

**GAE 上创建查询**

1. 登陆 `https://PROJECT_ID.appspot.com/admin`，验证账户后创建查询。
   1. Authorize Access > Successfully connected to Google Analytics > <kbd>Create Query</kbd>

2. Query
   1. GA Core Reporting API 查询请求可以在 [Query Explorer](https://ga-dev-tools.appspot.com/query-explorer/) 创建。
   2. 因为要查询的是 Pageviews:
      1. **start-date**: 博客发布首日。
      2. **end-dat**e: `today` (这是 GA Report 支持的参数，表示永远按当前查询日期为止）。
      3. **metrics**: `ga:pageviews`
      4. **dimensions**: `ga:pagePath`
      5. **filters**: `ga:pagePath!@=;ga:pagePath!@(。`
         1. 为了减少返回结果，减轻网络带宽，所以增加自定义过滤规则1：
         2. 其中 `;` 表示用 逻辑与 串联两条规则，`!@=` 表示`不含 =`，`!@(` 表示不含`(`。
   3. <kbd>Run Query</kbd>
   4. 拷贝 `API Query URI` 生成内容，填至 GAE 上 `SuperProxy` 的 `Encoded URI for the query` 即可。

3. <kbd>Save Query</kbd> [link](https://myochosite-291718.appspot.com/admin/query/manage?query_id=ahNwfm15b2Nob3NpdGUtMjkxNzE4chULEghBcGlRdWVyeRiAgIDo14eBCgw)
4. GAE 上保存查询后，会生成一个 Public Endpoint（公开的访问地址），用户访问它将返回 JSON 格式的查询结果。
5. 最后，在 Public Request Endpoint 点击 <kbd>Enable Endpoint</kbd> 使查询生效，

    ```
    Details about the configuration and the public URL for this query.
    Name	pageviewforblog
    URL	https://myochosite-291718.appspot.com/query?id=ahNwfm15b2Nob3NpdGUtMjkxNzE4chULEghBcGlRdWVyeRiAgIDo14eBCgw
    Formats	CSV  DataTable (JSON Response)  DataTable (JSON String)  JSON  TSV for Excel
    Status	Disabled
    API Request	https://www.googleapis.com/analytics/v3/data/ga?ids=ga%3A230544252&start-date=2020-01-01&end-date=today&metrics=ga%3Apageviews&dimensions=ga%3ApagePath&filters=ga%3ApagePath!%40%3D%3Bga%3ApagePath!%40(
    Owner	lgraceye@hotmail.com
    ```

6. Scheduling 中点击 <kbd>Start Scheduling</kbd> 开启定时任务。 [link](https://myochosite-291718.appspot.com/admin/query/manage?query_id=ahNwfm15b2Nob3NpdGUtMjkxNzE4chULEghBcGlRdWVyeRiAgIDo14eBCgw)

---

ref
- [1](https://taoalpha.github.io/blog/2015/06/07/tech-add-google-analytics-pageviews-to-jekyll-blog/)
- [jekyll-ga](https://github.com/developmentseed/jekyll-ga#readme)
- [2](http://zhangwenli.com/blog/2014/08/05/page-view-from-google-analytics-for-your-blog/)
- [3](https://devblog.dymel.pl/2016/10/13/public-api-for-your-google-analytics/)
- [4](https://blog.cotes.info/posts/fetch-pageviews-from-google-analytics/)
.
