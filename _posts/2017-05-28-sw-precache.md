---
layout: post
title: How does SW-Precache works?
author: "Hux"
header-style: text
lang: en
tags:
  - Web
  - PWA
  - En
---

[_SW-Precache_](https://github.com/GoogleChrome/sw-precache) _is a great Service Worker tool from Google. It is a node module designed to be_ _integrated_ _into your build process and to generate a service worker for you._ _Though_ _you can use sw-precache out of the box, you might still wonder what happens under the hood. There you go, this article is written for you!_

> This post was first published at [Medium](https://medium.com/@Huxpro/how-does-sw-precache-works-2d99c3d3c725)

## Overview

The core files involving in sw-precache are mainly three:

```
service-worker.tmpl  
lib/  
 ├ sw-precache.js  
 └ functions.js
```

`sw-precache.js` is the main entry of the module. It reads the configuration, processes parameters, populates the `service-worker.tmpl` template and writes the result into specified file. And`functions.js` is just a module containing bunch of external functions which would be all injected into the generated service worker file as helpers.

Since the end effect of sw-precache is performed by the generated service worker file in the runtime, a easy way to get an idea of what happens is by checking out source code inside `service-worker.tmpl` . It’s not hard to understand the essentials and I will help you.

## Initialization

The generated service worker file (let’s call it `sw.js` for instance) get configuration by text interpolation when `sw-precache.js` populating `service-worker.tmpl` .

```js
// service-worker.tmpl  
var precacheConfig = <%= precacheConfig %>;

// sw.js  
var precacheConfig = [  
  ["js/a.js", "3cb4f0"],   
  ["css/b.css", "c5a951"]  
]
```

It’s not difficult to see that it’s a list of relative urls and MD5 hashes. In fact, one thing that `sw-precache.js` do in the build time is to calculate hash of each file that it asked to “precache” from `staticFileGlobs` parameter.

In `sw.js`, `precacheConfig` would be transformed into a ES6 Map with structure `Map {absoluteUrl => cacheKey}` as below. Noticed that I omit the origin part (e.g. `http://localhost`) for short.

```js
> urlToCacheKeys  
< Map(2) {  
  "http.../js/a.js" => "http.../js/a.js?_sw-precache=3cb4f0",   
  "http.../css/b.js" => "http.../css/b.css?_sw-precache=c5a951"  
}
```

Instead of using raw URL as the cache key, sw-precache append a `_sw-precache=[hash]` to the end of each URL when populating, updating its cache and even fetching these subresouces. Those `_sw-precache=[hash]` are what we called **cache-busting parameter\***. It can prevent service worker from responding and caching out-of-date responses found in browsers’ HTTP cache indefinitely.

Because each build would re-calculate hashes and re-generate a new `sw.js` with new `precacheConfig` containing those new hashes, `sw.js` can now determine the version of each subresources thus decide what part of its cache needs a update. **This is pretty similar with what we commonly do when realizing long-term caching with webpack or gulp-rev, to do a byte-diff ahead of runtime.**

\*: Developer can opt out this behaviour with `dontCacheBustUrlsMatching` option if they set HTTP caching headers right. More details on [Jake’s Post](https://jakearchibald.com/2016/caching-best-practices/).

## On Install

> ServiceWorker gives you an install event. You can use this to get stuff ready, stuff that must be ready before you handle other events.

During the `install` lifecycle, `sw.js` open the cache and get started to populate its cache. One cool thing that it does for you is its **incremental update** mechanism.

Sw-precache would search each cache key (the values of `urlsToCacheKeys`) in the `cachedUrls`, a ES6 Set containing URLs of all requests indexed from current version of cache, and only `fetch` and `cache.put` resources couldn’t be found in cache, i.e, never be cached before, thus reuse cached resources as much as possible.

If you can not fully understand it, don’t worry. We will recap it later, now let’s move on.

## On Activate

> Once a new ServiceWorker has installed & a previous version isn’t being used, the new one activates, and you get an `activate` event. Because the old version is out of the way, it's a good time to handle schema migrations in IndexedDB and also delete unused caches.

During activation phase, `sw.js` would compare all existing requests in the cache, named `existingRequests` (noticed that it now contains resources just cached on installation phase) with `setOfExpectedUrls`, a ES6 Set from the values of `urlsToCacheKeys`. And delete any requests not matching from cache.

```js
// sw.js
existingRequests.map(function(existingRequest) {
  if (!setOfExpectedUrls.has(existingRequest.url)) {
    return cache.delete(existingRequest);
  }
})
```

## On Fetch

Although the comments in source code have elaborated everything well, I wanna highlight some points during the request intercepting duration.

### Should Respond?

Firstly, we need to determine whether this request was included in our “pre-caching list”. If it was, this request should have been pre-fetched and pre-cached thus we can respond it directly from cache.

```js
// sw.js*  
var url = event.request.url      
shouldRespond = urlsToCacheKeys.has(url);
```

Noticed that we are matching raw URLs (e.g. `http://localhost/js/a.js`) instead of the hashed ones. It prevent us from calculating hashes at runtime, which would have a significant cost. And since we have kept the relationship in `urlToCacheKeys` it’s easy to index the hashed one out.

_\* In real cases, sw-precache would take `ignoreUrlParametersMatching` and `directoryIndex` options into consideration._

### Navigation Fallback

One interesting feature that sw-precache provided is `navigationFallback`(previously `defaultRoute`), which detect navigation request and respond a preset fallback HTML document when the URL of navigation request did not exist in `urlsToCacheKeys`.

It is presented for SPA using History API based routing, allowing responding arbitrary URLs with one single HTML entry defined in `navigationFallback`, kinda reimplementing a Nginx rewrite in service worker\*. Do noticed that service worker only intercept document (navigation request) inside its scope (and any resources referenced in those documents of course). So navigation towards outside scope would not be effected.

_\* `navigateFallbackWhitelist` can be provided to limit the “rewrite” scope._

### Respond from Cache

Finally, we get the appropriate cache key (the hashed URL) by raw URL with `urlsToCacheKeys` and invoke `event.respondWith()` to respond requests from cache directly. Done!

```js
// sw.js*
event.respondWith(
  caches.open(cacheName).then(cache => {
    return cache.match(urlsToCacheKeys.get(url))
      .then(response => {
        if (response) return response;
      });
  })
);
```

_\* The code was “ES6-fied” with error handling part removed._

## Cache Management Recap

That’s recap the cache management part with a full lifecycle simulation.

### The first build

Supposed we are in the very first load, the `cachedUrls` would be a empty set thus all subresources listed to be pre-cached would be fetched and put into cache on SW install time.

```js
// cachedUrls  
Set(0) {}

// urlToCacheKeys  
Map(2) {  
  "http.../js/a.js" => "http.../js/a.js?_sw-precache=3cb4f0",   
  "http.../css/b.js" => "http.../css/b.css?_sw-precache=c5a951"  
}

// SW Network Logs  
[sw] GET a.js?_sw-precache=3cb4f0      
[sw] GET b.css?_sw-precache=c5a951
```

After that, it will start to control the page immediately because the `sw.js` would call `clients.claim()` by default. It means the `sw.js` will start to intercept and try to serve future fetches from caches, so it’s good for performance.

In the second load, all subresouces have been cached and will be served directly from cache. So none requests are sent from `sw.js`.

```js
// cachedUrls  
Set(2) {  
  "http.../js/a.js? _sw-precache=3cb4f0",   
  "http.../css/b.css? _sw-precache=c5a951"  
}

// urlToCacheKeys  
Map(2) {  
  "http.../js/a.js" => "http.../js/a.js? _sw-precache=3cb4f0",   
  "http.../css/b.js" => "http.../css/b.css? _sw-precache=c5a951"  
}

// SW Network Logs  
// Empty
```

### The second build

Once we create a byte-diff of our subresouces (e.g., we modify `a.js` to a new version with hash value `d6420f`) and re-run the build process, a new version of `sw.js` would be also generated.

The new `sw.js` would run alongside with the existing one, and start its own installation phase.

```js
// cachedUrls  
Set(2) {  
  "http.../js/a.js? _sw-precache=3cb4f0",   
  "http.../css/b.css? _sw-precache=c5a951"  
}

// urlToCacheKeys  
Map(2) {  
  "http.../js/a.js" => "http.../js/a.js? _sw-precache=d6420f",   
  "http.../css/b.js" => "http.../css/b.css? _sw-precache=c5a951"  
}

// SW Network Logs  
 [sw] GET a.js?_sw-precache=d6420f
```

This time, `sw.js` see that there is a new version of `a.js` requested, so it fetch `/js/a.js?_sw-precache=d6420f`  and put the response into cache. In fact, we have two versions of `a.js` in cache at the same time in this moment.

```js
// what's in cache?
http.../js/a.js?_sw-precache=3cb4f0
http.../js/a.js?_sw-precache=d6420f
http.../css/b.css?_sw-precache=c5a951
```

By default, `sw.js` generated by sw-precache would call `self.skipWaiting` so it would take over the page and move onto activating phase immediately.

```js
// existingRequests
http.../js/a.js?_sw-precache=3cb4f0
http.../js/a.js?_sw-precache=d6420f
http.../css/b.css?_sw-precache=c5a951

// setOfExpectedUrls
Set(2) {
  "http.../js/a.js?_sw-precache=d6420f", 
  "http.../css/b.css?_sw-precache=c5a951"
}

// the one deleted
http.../js/a.js?_sw-precache=3cb4f0
```

By comparing existing requests in the cache with set of expected ones, the old version of `a.js` would be deleted from cache. This ensure there is only one version of our site’s resources each time.

That’s it! We finish the simulation successfully.

## Conclusions

As its name implied, sw-precache is designed specifically for the needs of precaching some critical static resources. It only does one thing but does it well. I’d love to give you some opinionated suggestions but you decide whether your requirements suit it or not.

### Precaching is NOT free

So don’t precached everything. Sw-precache use a [“On Install — as a dependency”](https://jakearchibald.com/2014/offline-cookbook/#on-install-as-a-dependency) strategy for your precache configs. A huge list of requests would delay the time service worker finishing installing and, in addition, wastes users’ bandwidth and disk space.

For instance, if you wanna build a offline-capable blogs. You had better not include things like `'posts/*.html` in `staticFileGlobs`. It would be a huge disaster to data-sensitive people if you have hundreds of posts. Use a Runtime Caching instead.

### “App Shell”

> A helpful analogy is to think of your App Shell as the code and resources that would be published to an app store for a native iOS or Android application.

Though I always consider that the term “App Shell” is too narrow to cover its actual usages now, It is widely used and commonly known. I personally prefer calling them **“Web Installation Package”** straightforward because they can be truly installed into users’ disks and our web app can boot up directly from them in any network environments. The only difference between “Web Installation Package” and iOS/Android App is that we need strive to limit it within a reasonable size.

Precaching is perfect for this kinda resources such as entry html, visual placeholders, offline pages etc., because they can be static in one version, small-sized, and most importantly, part of critical rendering path. We wanna put first meaningful paint ASAP to our user thus we precache them to eliminate HTTP roundtrip time.

BTW, if you are using HTML5 Application Cache before, sw-precache is really a perfect replacement because it can cover nearly all use cases the App Cache provide.

### This is not the end

Sw-precache is just one of awesome tools that can help you build service worker. If you are planing to add some service worker power into your website, Don’t hesitate to checkout sw-toolbox, sw-helper (a new tool Google is working on) and many more from communities.

That’s all. Wish you enjoy!
