---
title: Implement Google Analytics and Page Views to blog
date: 2020-10-04 11:11:11 -0400
# description: IT Blog Pool
categories: [Note, GoogleAnalytics]
# img: /assets/img/sample/rabbit.png
img_path: /img/post/
tags:
---

# Implement Google Analytics and Page Views to blog

- [Implement Google Analytics and Page Views to blog](#implement-google-analytics-and-page-views-to-blog)
  - [Set up Google Analytics](#set-up-google-analytics)
    - [Create GA account and property](#create-ga-account-and-property)
    - [Create Data Stream](#create-data-stream)
  - [Setup Page Views](#setup-page-views)
    - [Setup Google App Engine](#setup-google-app-engine)
    - [Setup Google Analytics superProxy](#setup-google-analytics-superproxy)
    - [Create Google Analytics Query](#create-google-analytics-query)
  - [Configure Chirpy to Display Page View](#configure-chirpy-to-display-page-view)
  - [Google APIs 创建项目](#google-apis-创建项目)
  - [下载配置 SuperProxy](#下载配置-superproxy)

--- 

## Set up Google Analytics

### Create GA account and property

First, you need to set up your account on Google analytics. While you create your account, you must create your first **Property** as well.

1. Head to <https://analytics.google.com/> and click on **Start Measuring**
2. Enter your desired _Account Name_ and choose the desired checkboxes
3. Enter your desired _Property Name_. This is the name of the tracker project that appears on your Google Analytics dashboard
4. Enter the required information _About your business_
5. Hit _Create_ and accept any license popup to set up your Google Analytics account and create your property

### Create Data Stream

With your property created, you now need to set up Data Stream to track your blog traffic. After you signup, the prompt should automatically take you to create your first **Data Stream**. If not, follow these steps:

1. Go to **Admin** on the left column
2. Select the desired property from the drop-down on the second column
3. Click on **Data Streams**
4. Add a stream and click on **Web**
5. Enter your blog's URL

Now, click on the new data stream and grab the **Measurement ID**. It should look something like `G-V6XXXXXXXX`. Copy this to your `_config.yml`{: .filepath} file:

```yaml
google_analytics:
  id: 'G-V6XXXXXXX'   # fill in your Google Analytics ID
  # Google Analytics pageviews report settings
  pv:
    proxy_endpoint:   # fill in the Google Analytics superProxy endpoint of Google App Engine
    cache_path:       # the local PV cache data, friendly to visitors from GFW region
```


When you push these changes to your blog, you should start seeing the traffic on your Google Analytics. Play around with the Google Analytics dashboard to get familiar with the options available as it takes like 5 mins to pick up your changes. You should now be able to monitor your traffic in real time.


## Setup Page Views

There is a detailed [tutorial](https://developers.google.com/analytics/solutions/google-analytics-super-proxy) available to set up Google Analytics superProxy. But, if you are interested to just quickly get your Chirpy-based blog display page views, follow along. These steps were tested on a Linux machine. If you are running Windows, you can use the Git bash terminal to run Unix-like commands.

### Setup Google App Engine

1. Visit <https://console.cloud.google.com/appengine>

2. Click on **Create Application**

3. Click on **Create Project**

4. Enter the name and choose the data center close to you

5. Select **Python** language and **Standard** environment

6. Enable billing account. Yeah, you have to link your credit card. But, you won't be billed unless you exceed your free quota. For a simple blog, the free quota is more than sufficient.

7. Go to your App Engine dashboard on your browser and select **API & Services** from the left navigation menu

8. Click on **Enable APIs and Services** button on the top

9. Enable the following APIs: _Google Analytics API_

10. On the left, Click on _OAuth Consent Screen_ and accept **Configure Consent Screen**. Select **External** since your blog is probably hosted for the public. Click on **Publish** under _Publishing Status_

11. Click on **Credentials** on the left and create a new **OAuth Client IDs** credential. Make sure to add an entry under `Authorized redirect URIs` that matches: `https://<project-id>.<region>.r.appspot.com/admin/auth`

12. Note down the **Your Client ID** and **Your Client Secret**. You'll need this in the next section.

13. Download and install the cloud SDK for your platform: <https://cloud.google.com/sdk/docs/quickstart>

14. Run the following commands:

    ```console
    [root@bc96abf71ef8 /]# gcloud init

    ~snip~

    Go to the following link in your browser:

        https://accounts.google.com/o/oauth2/auth?response_type=code&client_id=XYZ.apps.googleusercontent.com&redirect_uri=ABCDEFG

    Enter verification code: <VERIFICATION CODE THAT YOU GET AFTER YOU VISIT AND AUTHENTICATE FROM THE ABOVE LINK>

    You are logged in as: [blah_blah@gmail.com].

    Pick cloud project to use:
    [1] chirpy-test-300716
    [2] Create a new project
    Please enter numeric choice or text value (must exactly match list
    item): 1


    [root@bc96abf71ef8 /]# gcloud info
    # Your selected project info should be displayed here
    ```

### Setup Google Analytics superProxy

1. Clone the **Google Analytics superProxy** project on Github: <https://github.com/googleanalytics/google-analytics-super-proxy> to your local.

2.  Remove the first 2 lines in the [`src/app.yaml`{: .filepath}](https://github.com/googleanalytics/google-analytics-super-proxy/blob/master/src/app.yaml#L1-L2) file:

    ```diff
    - application: your-project-id
    - version: 1
    ```

3. In `src/config.py`{: .filepath}, add the `OAUTH_CLIENT_ID` and `OAUTH_CLIENT_SECRET` that you gathered from your App Engine Dashboard.

4.  Enter any random key for `XSRF_KEY`, your `config.py`{: .filepath} should look similar to this

    ```python
    #!/usr/bin/python2.7

    __author__ = 'pete.frisella@gmail.com (Pete Frisella)'

    # OAuth 2.0 Client Settings
    AUTH_CONFIG = {
      'OAUTH_CLIENT_ID': 'YOUR_CLIENT_ID',
      'OAUTH_CLIENT_SECRET': 'YOUR_CLIENT_SECRET',
      'OAUTH_REDIRECT_URI': '%s%s' % (
        'https://chirpy-test-XXXXXX.ue.r.appspot.com',
        '/admin/auth'
      )
    }

    # XSRF Settings
    XSRF_KEY = 'OnceUponATimeThereLivedALegend'
    ```
    {: file="src/config.py"}

    > You can configure a custom domain instead of `https://PROJECT_ID.REGION_ID.r.appspot.com`.
    > But, for the sake of keeping it simple, we will be using the Google provided default URL.
    {: .prompt-info }

5.  From inside the `src/`{: .filepath} directory, deploy the app

    ```console
    [root@bc96abf71ef8 src]# gcloud app deploy
    Services to deploy:

    descriptor:      [/tmp/google-analytics-super-proxy/src/app.yaml]
    source:          [/tmp/google-analytics-super-proxy/src]
    target project:  [chirpy-test-XXXX]
    target service:  [default]
    target version:  [VESRION_NUM]
    target url:      [https://chirpy-test-XXXX.ue.r.appspot.com]


    Do you want to continue (Y/n)? Y

    Beginning deployment of service [default]...
    ╔════════════════════════════════════════════════════════════╗
    ╠═ Uploading 1 file to Google Cloud Storage                 ═╣
    ╚════════════════════════════════════════════════════════════╝
    File upload done.
    Updating service [default]...done.
    Setting traffic split for service [default]...done.
    Deployed service [default] to [https://chirpy-test-XXXX.ue.r.appspot.com]

    You can stream logs from the command line by running:
    $ gcloud app logs tail -s default

    To view your application in the web browser run:
    $ gcloud app browse
    ```

6. Visit the deployed service. Add a `/admin` to the end of the URL.

7. Click on **Authorize Users** and make sure to add yourself as a managed user.

8. If you get any errors, please Google it. The errors are self-explanatory and should be easy to fix.

If everything went good, you'll get this screen:

![superProxy](superProxy-deployed.png)

---

### Create Google Analytics Query

Head to `https://PROJECT_ID.REGION_ID.r.appspot.com/admin` and create a query after verifying the account. **GA Core Reporting API** query request can be created in [Query Explorer](https://ga-dev-tools.appspot.com/query-explorer/).

The query parameters are as follows:

- **start-date**: fill in the first day of blog posting
- **end-date**: fill in `today` (this is a parameter supported by GA Report, which means that it will always end according to the current query date)
- **metrics**: select `ga:pageviews`
- **dimensions**: select `ga:pagePath`

In order to reduce the returned results and reduce the network bandwidth, we add custom filtering rules [^ga-filters]:

- **filters**: fill in `ga:pagePath=~^/posts/.*/$;ga:pagePath!@=`.

  Among them, `;` means using _logical AND_ to concatenate two rules.

  If the `site.baseurl` is specified, change the first filtering rule to `ga:pagePath=~^/BASE_URL/posts/.*/$`, where `BASE_URL` is the value of `site.baseurl`.

After <kbd>Run Query</kbd>, copy the generated contents of **API Query URI** at the bottom of the page and fill in the **Encoded URI for the query** of SuperProxy on GAE.

After the query is saved on GAE, a **Public Endpoint** (public access address) will be generated, and we will get the query result in JSON format when accessing it. Finally, click <kbd>Enable Endpoint</kbd> in **Public Request Endpoint** to make the query effective, and click <kbd>Start Scheduling</kbd> in **Scheduling** to start the scheduled task.
 
## Configure Chirpy to Display Page View

Once all the hard part is done, it is very easy to enable the Page View on Chirpy theme. Your superProxy dashboard should look something like below and you can grab the required values.

![superproxy-dashboard](/posts/20210103/05-superproxy-dashboard.png){: width="1210" height="694"}

Update the `_config.yml`{: .filepath} file of [**Chirpy**][chirpy-homepage] project with the values from your dashboard, to look similar to the following:

```yaml
google_analytics:
  id: 'G-V6XXXXXXX'   # fill in your Google Analytics ID
  pv:
    proxy_endpoint: 'https://PROJECT_ID.REGION_ID.r.appspot.com/query?id=<ID FROM SUPER PROXY>'
    cache_path:       # the local PV cache data, friendly to visitors from GFW region
``` 

Now, you should see the Page View enabled on your blog.

![final](/assets/final.png)

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
