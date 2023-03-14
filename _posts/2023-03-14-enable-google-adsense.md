---
title: Enable Google Adsense
author: acdoussan
date: 2023-03-14 17:00:00 -0500
categories: [Blogging, Tutorial]
tags: [google adsense, adsense, ads, advertisements, advertisement, google]
---


This post is to enable Adsense on the [**Chirpy**][chirpy-homepage] theme based blog that you just built. This requires technical knowledge.

NOTE: If your blog isn't served at the root of your domain (aka www.example.com or example.com), adsense won't approve
your account. There is no way around this. If you want to set this up on a subdomain, serve your blog on both and
redirect www and no prefix to your blog while adsense is validating your domain. You can remove the redirect after, but
there is no guaruntee that they won't re-review and stop serving ads on your domain.

## Set up Google Adsense

See https://support.google.com/adsense/answer/7402253?hl=en for how to create an adsense account.

Navigate to the Sites tab on the left and enter in a domain for your blog, follow the prompts to create the site.
When you get the example code, save it or note down the client id and slot id.

## Configure Chirpy to serve ads

Update the `_config.yml`{: .filepath} file of [**Chirpy**][chirpy-homepage] project with the following values, use the
example code from creating the domain to fill in these values:

```yaml
google_adsense:
  client_id: ca-pub-0000000000000000
  slot_id: 8435997133
  enabled: true
```
{: file="_config.yml"}

Publish this change to your blog, and start the adsense verification process. Once your site is approved, visitors
should start getting shown ads!
## Reference

[chirpy-homepage]: https://github.com/cotes2020/jekyll-theme-chirpy/
