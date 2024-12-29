> Read at medium.com: [Upgrading Ele.me to Progressive Web Apps](https://medium.com/elemefe/upgrading-ele-me-to-progressive-web-app-2a446832e509)

Since the very first experiments that [@Vue.js tweeted][1], we at Ele.me (the biggest food ordering and delivering company in China) have been working on upgrading our mobile website to a [Progressive Web App][2]. We’re proud to ship the world-first PWA exclusively for the Chinese market, but even prouder to collaborate with Google, UC and Tencent to push the boundary of web experience and browser supports in China.


## Multi-page, Vue, PWA?

There is a prevailing opinion that only structuring a web app as a Single Page App can we build PWAs that deliver app-like user experience. Popular reference examples including [Twitter Lite][3], [Flipkart Lite][4], [Housing Go][5] and [Polymer Shop][6] are all using the SPA model.

However at Ele.me, we’ve come to appreciate many advantages of a Multi-Page App model, and decided to refactor the mobile site from an Angular 1 SPA to a Multi-Paged app more than a year ago. The most important advantage we see is the isolation and decoupling between pages, which allows us to built different parts of the mobile site as “micro-services”. These services can then be independently iterated, embedded into 3rd-party apps, and even maintained by different teams.

Meanwhile, we still leverage [Vue.js](http://vuejs.org/) to boost our productivity. You may have heard of Vue as a rival of React or Angular, but Vue's lightweight and performance make it also a perfect replacement of traditional "jQuery/Zepto + template engine" stack when engineering a Multi-page app. We built every component as [Single File Components](http://vuejs.org/v2/guide/single-file-components.html) so they can be easily shareable between pages. The declarative-ness plus reactivity Vue offered help us manage both code and data flow. Oh, did I mention that [Vue is progressive](https://www.youtube.com/watch?v=pBBSp_iIiVM)? So things like Vuex or Vue-Router can be incrementally adopted if our site's complexity scales up, like...migrating to SPA again? (Who knows...)
 
In 2017, PWA seems to be all the rage, so we embark on exploring how far can our Vue-based Multi-page PWAs actually go.

## Implementing "PRPL" with MPA

I love [PRPL pattern][7] because it gives you a high-level abstraction of how to structure and design your own PWA systems. Since we are not rebuild everything from scratch, we decided taking implementing PRPL as our migration goal:

### 1. PUSH critical resources for initial URL route. 

The key of pushing/preloading is to prioritize resources hidden in deep dependency graph and make browser's network stack busy ASAP. Let's say you have a SPA with code splitting by route, you can push/preload chunks for the current route before the "entry chunks" (e.g. webpack manifest, router) finish downloading and evaluating. So when the actual fetches happen, they might already be in caches.

Routes in MPAs naturally fetch code for that route only, and tend to have a flattening dependency graph. Most scripts depended by Ele.me are just `<script>` elements, so they can be found and fetched by [good old browser preloader][8] in early parsing phase without explicit `<link rel="preload">`. 

![](/img/in-post/post-eleme-pwa/PUSH-link-rel-preload.jpg)

To take benefits from HTTP2 Multiplexing, we currently serve all critical resources under a single domain (no more domain sharding), and we are also experimenting on Server Push.


### 2. RENDER initial route & get it interactive ASAP

This one is essentially free (ridiculously obvious) in MPA since there's only one route at one time.

A straightforward rendering is critical for metrics such as First-Meaningful-Paint and Time-To-Interactive. MPAs gain it for free due to the simplicity of traditional HTML navigation they used. 

### 3. **PRE-CACHE** remaining routes using Service Worker

This's the part [Service Worker][9] come to join the show. Service Worker is known as a client-side proxy enabling developers to intercept requests and serve responses from cache, but it can also perform initiative fetch to prefetch then precache future resources.

![](/img/in-post/post-eleme-pwa/PRECACHE-future-routes.jpg)

We already used [Webpack][10] in the build process to do `.vue` compilation and asset versioning, so we create a webpack plugin to help us collecting dependencies into a "precache manifest" and generating a new Service Worker file after each build. This is pretty much like [how SW-Precache works][11]. 

**In fact, we only collect dependencies of routes we flagged as "Critical Route".** You can think of them as ["App Shell"][12] or the "Installation Package" of our app. Once they are cached/installed successfully, our web app can boot up directly from cache and available offline. Routes that "not critical" would be incrementally cached at runtime during the first visit. Thanks to the LRU cache policies and TTL invalidation mechanisms provided by [SW-Toolbox][13], we have no worries of hitting the quota in a long run.

### 4. LAZY-load & instantiate remaining routes on demand 

Lazy-loading and lazily instantiating remaining parts of the app is relatively challenging for SPA to achieve. It requires both code splitting and async importing. Fortunately, this is also a built-in feature of MPA model, in which routes are naturally separated.

Noticed that the lazy-loading can be done instantly if the requested route is already pre-cached in Service Worker cache, no matter whether SPA or MPA is used. #ServiceWorkerAwesomeness 


---

Surprisingly, we found Multi-page PWA is kinda naturally "PRPL"! MPA has already provided built-in support for "PRL", and the second "P" involving Service Worker can be easily fulfilled in any PWA. 

So what about the end result?


![](/img/in-post/post-eleme-pwa/Lighthouse-before.png)

**In [Lighthouse](https://developers.google.com/web/tools/lighthouse/) simulation (3G & 5x Slower CPU), we made Time-To-Interactive around 2 seconds,** and this was benchmarked on our HTTP1 test server. 

The first visit is fast. The repeat visit with Service Worker is even faster. You can check out this video to see the huge difference between with or without Service Worker:

<iframe width="560" height="315" src="https://www.youtube.com/embed/mbi_WnunJa8" frameborder="0" allowfullscreen></iframe>

Did you see that? No, I mean the annoying blank screen. Even in the Service Worker one, the blank screen is still conspicuous during navigating. How can that be?


## Multi-page Pitfall: Redo Everything!

Unlike SPA, changing routes in MPA means actual browser navigation happens: The previous page is discarded completely and the browser need to redo everything for next route: re-download resources, re-parse HTML, re-evaluate JavaScript, re-decode image data, re-layout the page and re-paint the screen, even many of them could be shared across routes. All of these works combined requires significant computing and time.

So here is the profile (2x slower CPU simulated) of our entry page (most heavy one). Even we can make Time-To-Interactive around 1s in repeat visit, our users can still feel too slow for just "switching a tab".

![](/img/in-post/post-eleme-pwa/msite-Before-Optim.png)

### Huge JavaScript Re-Startup Cost

According to the profile, most of the time (900ms) before hitting the first paint is spent on evaluating JavaScript. Half is on dependencies including Vue Runtime, components, libraries etc., another half is on actual Vue starting-up and mounting. Because all UI rendering is depended on JavaScript/Vue, all of the critical scripts remain guiltily parser-blocking. I'm by no means blaming JavaScript or Vue's overheads here, It's just a tradeoff when we need this layer of abstraction in engineering.

**In SPA, JavaScript Start-up Cost is amortized during the whole lifecycle.** Parsing/Compiling for each script is only once, many heavy executing can be only once. The big JavaScript objects like Vue's ViewModels and Virtual DOM can be kept in memory and reused as much as you want. **This is not the case in MPA however.**


### Could Browser Caches Help?

Yes or no.

V8 introduced [code caching](http://v8project.blogspot.com/2015/07/code-caching.html), a way to store a local copy of compiled code so fetching, parsing and compilation could all be skipped next time. As @addyosmani mentioned in [JavaScript Start-up Performance](https://medium.com/reloading/javascript-start-up-performance-69200f43b201), scripts stored in Cache Storage via Service Worker could trigger code caching in just the first execution.

Another browser cache you might hear of is "Back-Forward Cache", or bfcache. The name varies, like Opera's "Fast History Navigation" or [WebKit's "Page Cache"](https://webkit.org/blog/427/webkit-page-cache-i-the-basics/). **The idea is that browsers can keep the previous page live in memory, i.e. DOM/JS states, instead of destroying everything.** In fact, this idea works very well for MPA. You can try every traditional Multi-page websites in iOS Safari and observe an instantaneously loading when back/forward. (With browser UI/Gesture or with hyperlink can have a slight difference though.)

Unfortunately, Chrome has no this kind of in-memory bfcache currently concerning to memory consumption and its multi-process architecture. It just leverages HTTP disk cache to simplify the loading pipeline, almost everything still needs to be redone. More details and discussions can be seen [here](https://docs.google.com/document/d/1o8KImLPrJQcMNqvd_a-1V8fEVgtVeEJww453ZQ1hGuo/edit#).




## Striving for Perceived Performance

Although the reality is dark, we don't want to give up so easily. One optimization we try to do is to render DOM nodes/create Virtual DOM nodes as less as possible to improve the Time-To-Interactive. While another opportunity we see is to play tricks on perceived performance.

@owencm have written a great post ["Reactive Web Design: The secret to building web apps that feel amazing"](https://medium.com/@owencm/reactive-web-design-the-secret-to-building-web-apps-that-feel-amazing-b5cbfe9b7c50) covering both "Instant loads with skeleton screens" and "Stable loads via predefined sizes on elements" to improve perceived performance and user experience. Yes, we actually used both. 

What about we showing the end result after these optimizations first before entering technical nitty gritty? There you go!

<iframe width="560" height="315" src="https://www.youtube.com/embed/K5JBGnMYO1s" frameborder="0" allowfullscreen></iframe>

Too fast and can not see the pulsing Skeleton Screen clearly? Here is a version showing how it looks like under 10 times slower CPU.

<iframe width="560" height="315" src="https://www.youtube.com/embed/w1ZbNsHmRjs" frameborder="0" allowfullscreen></iframe>

This is a much better UX, right? Even we have slow navigation in slow devices, at least the UI is stable, consistent and always responding. So how we get there?



### Rendering Skeleton Screen with Vue at Build-Time 

As you might have guessed, the Skeleton Screen that consists of markups, styles, and images is inlined into `*.html` of each route. So they can be cached by Service Worker, be loaded instantly, and be rendered independently with any JavaScript.

We don't want to manually craft each Skeleton Screen for each routes. It's a tedious job and we have to manually sync every change between Skeleton Screens and the actual UI components (Yes we treat every route as just a Vue component). But think about it, [Skeleton Screen is just a blank version of a page into which information is gradually loaded](https://www.lukew.com/ff/entry.asp?1797). What if we bake the Skeleton Screen into the actual UI component as just a loading state so we can render Skeleton Screen out directly from it without the issue of syncing?

Thanks to the versatility of Vue, we can actually realize it with [Vue.js Server-Side Rendering](https://ssr.vuejs.org/en/). Instead of using it on a real server, we use it at build time to render Vue components to strings and injected them into HTML templates.



### Fast Skeleton Painting...

Having markups in `*.html` doesn't mean that they will be painted fast, you have to make sure the [Critical Rendering Path](https://developers.google.com/web/fundamentals/performance/critical-rendering-path/) is optimized for that. Many developers believed that putting script tags in the end of the body is sufficient for getting content painted before executing scripts. This might be true for browsers supporting rendering an incomplete DOM tree (e.g. streaming render), But browsers might not do that in mobile concerning slower hardwares, battery, and heats. **And even we are told that script tags with `async` or `defer` is not parser-blocking, it also doesn't mean we can get content painted before executing scripts in reality.**

![](https://html.spec.whatwg.org/images/asyncdefer.svg)

First I want to clarify it a little bit. According to the [Scripting section of HTML](https://www.w3.org/TR/html51/semantics-scripting.html#elementdef-script) (WHATWG living standard, the W3C's same here), `async` scripts would be evaluated as soon as it is available thus could potentially blocking parsing. Only `defer` (and not inlined) is specified to be never block parsing. That's why [Steve Souders](http://stevesouders.com/) ever posted ["Prefer DEFER Over ASYNC"](https://calendar.perfplanet.com/2016/prefer-defer-over-async/). (`defer` has its own issue and we will cover it later.)

**Then I want to say: A script not blocking parser could still block painting nonetheless.** So here is a reduced test I wrote named **"Minimal Multi-page PWA"**, or MMPWA, which basically render 1000 list items within an `async` (and truly not parser-blocking) script to see if we can get Skeleton Screen painted before scripts get executed. The profile below (over USB debugging on my real Nexus 5) shows my ignorance:

![](/img/in-post/post-eleme-pwa/thisTick-&-Load.png)

Yes, keep your mouth open. The first paint is blocked. I was also surprised here. The reason I guess is that **if we touch DOM so quickly that the browser has still NOT finished previous painting job, our dear browser has to abort every pixel it has drawn, and has to wait until current DOM manipulation task finished and redo the rendering pipeline again.** And this more often happens with a mobile device with a slower CPU/GPU.

### Fast Skeleton Painting with setTimeout Hack

We indeed encountered this problem when testing our new beautiful Skeleton Screen. Perhaps Vue finishes its job and start to mount nodes too fast ;). But anyway we have to make it slower, or rather lazier. So we try to put DOM manipulation things inside `setTimeout(callback, 0)`, and it works like a charm!

![](/img/in-post/post-eleme-pwa/nextTick-&-Load.png)


I think you may curious about how this change performs in the wild, so I have refined MMPWA by rendering 5000 list items rather 1000 to make the differences more obvious, and by designing it in an A/B testing manner. The code is on [Github](https://github.com/Huxpro/mmpwa) and the demo is live on [huangxuan.me/mmpwa/](https://huangxuan.me/mmpwa). Here is also a video for loungers.


<iframe width="560" height="315" src="https://www.youtube.com/embed/3Ws7XBHrPD8" frameborder="0" allowfullscreen></iframe>


This famous `setTimeout` hack (a.k.a. Zero Delays) looks quite magic,  but it is science™. If you are familiar with **event loop**, it just prevents these code from executing in the current loop by putting everything to the **task queues** with the Timer Callback, so the browser could breath (update the rendering) in the main thread. 


So we applied what we learned from MMPWA by putting `new Vue()` inside `setTimeout` and BOOM! We have Skeleton Screen painted consistently after every navigating! Here is the profile after all these optimizations.

![](/img/in-post/post-eleme-pwa/msite-After-Optim.png)

Huge improvements right?  This time we hit First Paint (Skeleton Screen Paint) at 400ms and TTI at 600ms. You should really go back to have a before-after comparison in details.

### One more thing that I deferred

But wait, why is there still a bunch of guiltily parser-blocking scripts? Are them all `async`? OK, ok. For historical reasons, we do keep some parser-blocking scripts, like [lib-flexible](https://github.com/amfe/lib-flexible), we couldn't get rid of it without a huge refactoring. But most of these blocking scripts are in fact `defer`. We expected that they can be executed after parsing and in order, however the profile kinda slap on my face. :(

Remember I said I would talk about one issue of `defer` previously? Yes, that's it. I have had a [conversation](https://twitter.com/Huxpro/status/859842124849827841) with [Jake Archibald](https://twitter.com/jaffathecake) and it turns out it might be a bug of Chrome when the deferred scripts are fully cached. [Vote it at crbug](https://bugs.chromium.org/p/chromium/issues/detail?id=717979)!


Similar improvements can be seen from Lighthouse (Under same server and network environment). A Pro Tip is you should always use lighthouse in a variable controlling approach.

![](/img/in-post/post-eleme-pwa/Lighthouse-after.png)


### Performance In the Real World

[Alex Russell](https://medium.com/@slightlylate) has given [a very insightful talk on mobile web performance](https://youtu.be/4bZvq3nodf4?list=PLNYkxOF6rcIBTs2KPy1E6tIYaWoFcG3uj) at Chrome Dev Summit 2016, talking about how hard can we build performant web applications on mobile devices. Highly recommended.

Chinese users tend to have a pretty powerful phone. MI4 is shipped with snapdragon 801 (slightly out-performs Nexus 5) but only costs 100$. It’s affordable by at least 80% of our users so we take it as a baseline.

Here is a video screen-recorded on my Nexus 5 showing switching between 4 tabs. The performance varies between tabs due to their variant scale. The heaviest one, entry page, take around 1s to hit real Time-To-Interactive on my Nexus 5.

FYI. This is surprisingly comparable to what I get from Chrome Simulation with 2x CPU throttling. With 5x throttling, this can spend 2–3s to get TTI, horribly. (To be honest, I found even under same throttling, the results can vary drastically depended on my Macbook’s “mood”.)

<iframe width="700" height="525" src="https://www.youtube.com/embed/ZLc8jysMqaw?ecver=1" frameborder="0" allowfullscreen></iframe>


## Final Thoughts


This article is much longer than I could imagine. I am really appreciated if you could get here. So what can we learn from it?

### MPA still has some way to go

[Jake Archibald](https://twitter.com/jaffathecake) ever said that "PWA !== SPA" at [Chrome Dev Summit 2016](https://youtu.be/J2dOTKBoTL4?list=PLNYkxOF6rcIBTs2KPy1E6tIYaWoFcG3uj). But the sad truth is that even we have taken advantages of bleeding edge technologies such as "PRPL" pattern, Service Worker, App-Shell, Skeleton Screen, there is still a distance between us and many Single Page PWA just because we are Multi-page structured.

The web is extremely versatile. Static blogs, e-business sites, desktop-level software, all of them should be the first-class citizens of the web family. MPA might have things like "bfcache API", navigation transitions to catch up the SPA in the future, but it is not today certainly. 


### PWA is Awesome No Matter What

Hey, I am not overblowing it. Even we as a Multi-page PWA couldn't be as stunning and app-like as many Single Page PWAs are. The idea and technologies behind PWA still help us deliver a much better experience to our users on the web that hasn’t been possible before.

What PWA is trying to solve are some fundamental problems of current web application model such as its hard dependencies to network and browser UIs. That' why PWA can be always beneficial no matter what architecture or what framework you actually used. [Addy Osmani](https://medium.com/@addyosmani) would give a talk [Production Progressive Web Apps With JavaScript Frameworks](https://events.google.com/io/schedule/?section=may-19&sid=e8436b55-ea89-4243-a644-5ecb319d9ef0) at this year's I/O (and [I/O 16](https://youtu.be/srdKq0DckXQ?list=PLNYkxOF6rcIDz1TzmmMRBC-kd8zPRTQIP)). You won’t want to miss it!

---

Finally, I’d love to thank:

- my colleagues [YiSi Wang](https://github.com/YiSiWang), [GuangHui Ren](https://github.com/rguanghui), [JiyinYiyong](https://medium.com/@jiyinyiyong) from Eleme
- collaborator [Michael Yeung](https://medium.com/@micyeung), [Liam Spradlin](https://medium.com/@LiamSpradlin) and other collaborators from Google
- collaborators from UC/Tencent

And special thanks to 

- invited reviewer, [Evan You](https://medium.com/@youyuxi).
- Chrome “StackOverflow”, [Jake Archibald](https://twitter.com/jaffathecake).

Thank you all!

---

## Appendix. Architecture Diagram

![](/img/in-post/post-eleme-pwa/Architecture.png)



[1]: https://twitter.com/vuejs/status/834087199008239619
[2]: https://developers.google.com/web/progressive-web-apps/
[3]: https://blog.twitter.com/2017/how-we-built-twitter-lite
[4]: https://medium.com/progressive-web-apps/building-flipkart-lite-a-progressive-web-app-2c211e641883
[5]: https://medium.com/engineering-housing/progressing-mobile-web-fac3efb8b454
[6]: https://shop.polymer-project.org/
[7]: https://developers.google.com/web/fundamentals/performance/prpl-pattern/
[8]: https://calendar.perfplanet.com/2013/big-bad-preloader/
[9]: https://w3c.github.io/ServiceWorker/v1/
[10]: https://webpack.github.io/
[11]: https://medium.com/@Huxpro/how-does-sw-precache-works-2d99c3d3c725
[12]: https://developers.google.com/web/updates/2015/11/app-shell
[13]: https://googlechrome.github.io/sw-toolbox/
