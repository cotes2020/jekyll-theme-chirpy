---
title: Enable Google Page Views
author: Dinesh Prasanth Moluguwan Krishnamoorthy
date: 2021-01-03 18:32:00 -0500
categories: [Blogging, Tutorial]
tags: [google analytics, pageviews]
---


This post is to enable Page Views on the [**Chirpy**](https://github.com/cotes2020/jekyll-theme-chirpy/fork) theme based blog that you just built. This requires technical knowledge and it's recommended to keep the `google_analytics.pv` disabled unless you have a good reason. If you website has low traffic, the Page View count would discourage you to write more blogs. With that said, let's start with the setup.

# Set up Google Analytics

## Create GA account and property
First, you need to setup your account on Google analytics. While your create your account, you must create your first **Property** as well.

1. Head to <https://https://analytics.google.com/> and click on **Start Measuring**
2. Enter your desired *Account Name* and choose the desired checkboxes
3. Enter your desired *Property Name*. This is the name of the tracker project that appears on your Google Analytics dashboard
4. Enter the required information *About your business*
5. Hit *Create* and accept any license popup to setup your Google Analytics account and create your property


## Create Data Stream

With your property created, you now need to set up Data Stream to track your blog traffic. After you signup, the prompt should automatically take you to creating your first **Data Stream**. If not, follow these steps:

1. Go to **Admin** on the left column
2. Select the desired property from the drop down on the second column
3. Click on **Data Streams**
4. Add a stream and click on **Web**
5. Enter your blog's URL

It should look like this:
![Desktop View](/assets/img/sample/01-google-analytics-data-stream.png)

Now, click on the new data stream and grab the **Measurement ID**. It should look something like `G-V6XXXXXXXX`. Copy this to your `_config.yaml` file

```yaml
google_analytics:
  id: 'G-V6XXXXXXX'          # Fill with your Google Analytics ID
  pv:
    # The Google Analytics pageviews switch.
    # DO NOT enable it unless you know how to deploy the Google Analytics superProxy.
    enabled: false
    # the next options only valid when `google_analytics.pv` is enabled.
    proxy_url: ''
    proxy_endpoint: ''
    cache: false  # pv data local cache, good for the users from GFW area.
```

When you push these changes to your blog, you should start seeing the traffic on your Google Analytics. Play around with Google Analytics dashboard to get familiar with the options available as it takes like 5 mins to pickup your changes. You should now be able to monitor your traffic in realtime.

![Desktop View](/assets/img/sample/02-google-analytics-realtime.png)

# Setup Page Views

There is a detailed [tutorial](https://developers.google.com/analytics/solutions/google-analytics-super-proxy) available to set up Google Analytics superProxy. But, if you are interested to just quickly get your Chirpy-based blog display page views, follow along. These steps were tested on a Linux machine. If you are running windows, you can use Git bash terminal to run linux-like commands.

## Setup Google App Engine

1. Visit <https://console.cloud.google.com/appengine>
2. Click on **Create Application**
3. Click on **Create Project**
4. Enter the name and choose the data center close to you
5. Select **Python** language and **Standard** environment
6. Enable billing account. Yeah, you have to link your credit card. But, you won't be billed unless you exceed your free quota. For a simple blog, free quota is more than sufficient.
7. Go to your App Engine dashboard on your browser and select **API & Services** from the left navigation menu
8. Click on **Enable APIs and Services** button on the top
9. Enable the following APIs: *Google Analytics API*
10. On the left, Click on *OAuth Consent Screen* and accept **Configure Consent Screen**. Select **External** since your blog is probably hosted for the public. Click on **Publish** under *Publishing Status*
11. Click on **Credentials** on the left and create a new **OAuth Client IDs** credential. Make sure to add a entry under `Authorized redirect URIs` that matches: `https://<project-id>.<region>.r.appspot.com/admin/auth`
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

## Setup Google Analytics superProxy


1. Clone the **Google Analytics superProxy** project on Github: <https://github.com/googleanalytics/google-analytics-super-proxy> to your local
2. Remove the first 2 lines in the [`src/app.yaml`](https://github.com/googleanalytics/google-analytics-super-proxy/blob/master/src/app.yaml#L1-L2) file

    ```yaml
    runtime: python27
    api_version: 1
    threadsafe: yes
    ...
    ```

3. In [`src/config.py`](), add the `OAUTH_CLIENT_ID` and `OAUTH_CLIENT_SECRET` that you gathered from you App Engine Dashboard
4. Enter any random key for `XSRF_KEY`

Your `config.py` should look similar to this:

    #!/usr/bin/python2.7
    #
    # ~snip~

    __author__ = 'pete.frisella@gmail.com (Pete Frisella)'

    # OAuth 2.0 Client Settings
    AUTH_CONFIG = {
        'OAUTH_CLIENT_ID': 'YOUR_CLIENT_ID',
        'OAUTH_CLIENT_SECRET': 'YOUR_CLIENT_SECRET',

        # E.g. Local Dev Env on port 8080: http://localhost:8080
        # E.g. Hosted on App Engine: https://your-application-id.appspot.com
        'OAUTH_REDIRECT_URI': '%s%s' % (
            'https://chirpy-test-XXXXXX.ue.r.appspot.com',
            '/admin/auth')
    }

    # XSRF Settings
    XSRF_KEY = 'OnceUponATimeThereLivedALegend'

**Tip:** You can configure a custom domain instead of <https://PROJECT_ID.REGION_ID.r.appspot.com>. But, for the sake of keeping it simple, we will be using the Google provided default URL.


5. From inside the src/ directory, deploy the app

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
8. If you get any errors, please google it. The errors are self-explanatory and should be easy to fix.

If everything went good, you'll get this screen:

![DesktopView](/assets/img/sample/03-superProxy-deployed.png)

## Create Google Analytics Query

*Create a Google Analytics Query*

# Configure Chirpy to display Page View

Once all the hard part is done, it is very easy to enable the Page View on Chirpy theme. Your superProxy dashboard should look something like below and you can grab the required values.

![Desktop View](/assets/img/sample/04-superproxy-dashboard.png)

Update the `_config.yml` file with the values from your dashboard, to look similar to the following:

    ```yaml
    google_analytics:
        id: 'G-XXXXXXXXXX'           # Fill with your Google Analytics ID
        pv:
            # The Google Analytics pageviews switch.
            # DO NOT enable it unless you know how to deploy the Google Analytics superProxy.
            enabled: true
            # the next options only valid when `google_analytics.pv` is enabled.
            proxy_url: 'https://PROJECT_ID.REGION_ID.r.appspot.com'
            proxy_endpoint: 'https://PROJECT_ID.REGION_ID.r.appspot.com/query?id=<ID FROM SUPER PROXY>'
            cache: false  # pv data local cache, good for the users from GFW area.
    ```

Now, you should see the Page View enabled on your blog.