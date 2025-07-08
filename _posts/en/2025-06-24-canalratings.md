---
title: "Canal+ Ratings : Firefox extension"
description: "Adds IMDb, Rotten Tomatoes and Allocine ratings on Canal+ streaming platform"
date: 2025-06-24
categories: [Website]
tags: [website, firefox, extension, javascript]
media_subpath: /assets/img/posts/canalratings
lang: en
image: 
    path: cover.png
---

Finding something to watch on a streaming platform like Canal+ (French service) can sometimes be overwhelming. With so many movies and TV shows available, it’s not always easy to decide what’s worth the time. The built-in rating system of the platform isn’t very reliable, so I often ended up checking ratings manually on sites like IMDb or Rotten Tomatoes. 

To simplify this process, I decided to build a small Firefox extension that automatically fetches ratings from multiple sources and displays them directly inside Canal+. This way, it becomes much quicker to spot the best movies without leaving the platform.

> The following project have been written in vanilla Javascript and the source code can be found here : [https://github.com/nicopaulb/Canal-Ratings](https://github.com/nicopaulb/Canal-Ratings).
{: .prompt-tip }

## Ratings sources
The first step was to identify which ratings would be the most useful to display. From experience, I selected three well-known platforms:  

- **IMDb** – Probably the most recognized international movie database. Its rating system is based on votes from millions of users around the world, giving a global and community-driven score.  
- **Rotten Tomatoes** – A rating aggregator that distinguishes between critics and audiences. The Tomatometer reflects professional critic reviews, while the Audience Score shows what general viewers think. This dual perspective is often very helpful.  
- **Allociné** – The largest French cinema website, combining an extensive movie catalog, editorial reviews, and audience ratings. Since Canal+ has a strong French catalog, Allociné felt like a natural choice. 

These three sources provide a good balance between **international opinions**, **professional reviews**, and **local French references**.  

### API

To automatically retrieve ratings for a specified movie, each of the selected websites offers an API. However, for a small personal project like this, official APIs are often too expensive or restricted. 
As an example, accessing IMDb data through their official API is available via AWS Data Exchange, with prices starting at $150,000 per year. Clearly, this is far beyond the budget of a personal project.

Given these limitations, relying on official APIs was not a viable option. Instead, I implemented a **scraping solution** to fetch the ratings directly from the web.

### Scraping
The most obvious solution would be to scrape the web pages of each rating website individually. However, this approach is time-consuming and difficult to maintain, because each source has its own layout and anti-scraping measures, such as rate limits and frequently changing page structures.  

I realized there was a simpler alternative: when you search for a movie on a search engine like Google, the overview often includes ratings from multiple sources, all in one place. 

![Search Result Showing Movie Ratings](search_results.png)
_Google search result showing ratings_

So instead of parsing each website separately, I decided to parse the search engine results to collect ratings from IMDb, Rotten Tomatoes, and Allociné simultaneously.  

To improve reliability, I implemented this approach across multiple **search engines**: Google, Bing, and Yahoo. This provides a backup if one search or scraping attempt fails, and increases the likelihood of retrieving ratings from all three sources, since some search engines may display only a subset of the ratings.

### Handling request limits  

Of course, search engines also have **anti-scraping measures**, such as detecting when too many requests are sent from the same IP. For a single movie, this is not a problem, but when fetching ratings for hundreds of movies at once (e.g., to display ratings directly on all thumbnails), it becomes an issue.  

![Google Captcha](captcha.png)
_Google Captcha_

The first solution is to implement a **caching system**. Movie ratings do not change frequently, so once fetched, they can be stored and reused without sending new requests every time. The caching duration is configurable (and can even be disabled, though not recommended), and is set by default to 7 days (configurable in [Option pages](#option-page)). All cached data is stored in the Firefox extension storage, making it persistent across browser restarts and system reboots.  

Caching alone is not enough. When many ratings need to be fetched for new movies, I also added a small delay between requests and prevented multiple requests from being sent in parallel. This avoids triggering the search engines’ bot detection mechanisms. To manage this, I implemented a **queue system**: all outgoing requests are stored in a queue, and requests to the same search engine are spaced by a configurable interval. This ensures smooth operation while minimizing the risk of being blocked.

I also considered using **proxies** to make requests from different IP addresses, but this proved difficult to implement in a Firefox extension, likely due to security restrictions. Fortunately, the current system works well in most cases, and with multiple search engines in place, if a request fails on one engine, the others serve as reliable backups.

## Customizing the Canal+ page 

Now that the extension can automatically fetch ratings for any movie, the next step was to integrate this functionality directly into the Canal+ interface. To make the extension work, two main tasks needed to be addressed:  

1. **Extracting the movie or TV show name** from the page’s HTML, so the extension knows which title to search for.  
2. **Displaying the fetched ratings** on the same page, both on the detail view and on thumbnails, for a seamless user experience.  

The first step was to implement these features on the **movie detail page** (the page you see after clicking on a movie). This was simpler because only one movie’s rating needs to be displayed. I decided to place the ratings just below the official Canal+ ratings using the HTML DOM.

![Movie Details Ratings Page](ratings_details.png)
_Ratings on movie details page_

Once this worked, I extended the functionality to **movie thumbnails** in the main page and across various carousels. This was more challenging because the movies are loaded dynamically. To handle this, I used a **MutationObserver** to detect new elements being added to the DOM. The ratings are displayed in the **bottom-right corner of each thumbnail**, inside a slightly transparent box. This way, it’s possible to quickly see and compare the ratings for all movies directly from the main page. Additionally, the thumbnail ratings can be disabled via the extension options for users who prefer a cleaner interface. 

![Movie Thumbnails Ratings Page](ratings_thumbnails.png)
_Ratings on movie thumbnails_

With the introduction of ratings on thumbnails, the number of requests increased significantly, leading to a larger queue. Fetching ratings for all movies on the current page can take up to 30 seconds, and even minutes if the user navigates quickly and adds new movies to the queue (assuming the cache is empty). This is a necessary trade-off to avoid triggering bot detection on search engines (see [Handling request limits](#handling-request-limits)).  

To improve user experience, I implemented a **priority system** in the queue. If a user clicks on a movie whose rating has not yet been fetched because it is far down in the queue, this movie is automatically moved to the front. This ensures that the rating for the selected movie is displayed immediately, without waiting for all other ratings to be processed first.  

Another important improvement was handling movies that fail to fetch a rating. This can happen if a movie title is too generic or if the movie is not widely recognized by search engines. In such cases, the failure is recorded to prevent repeated unnecessary requests, which would slow down the queue. Instead, a retry is scheduled after a configurable interval (1 hour by default, adjustable in the [Option pages](#option-page)).  

Finally, to keep users informed, a **loading spinner** is displayed in place of the rating while it is being fetched. This provides visual feedback that the extension is actively working and that the rating will appear shortly.  

![Movie Thumbnails Loading](spinner.png)
_Ratings loading on a movie thumbnail_

## Bundle a Firefox extension

I chose to publish the extension on the **official Firefox Add-ons Store** to familiarize myself with the process. Thanks to the detailed [Mozilla documentation](https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/Your_first_WebExtension), the procedure was straightforward. 

The main effort went into creating the `manifest.json` file, which includes the extension description, list of source files, required permissions, and the extension ID. Once registered, the extension became quickly available on the Firefox Store.

> Add-ons is available here for Firefox : [Movie Ratings for Canal+](https://addons.mozilla.org/fr/firefox/addon/movie-ratings-for-canal/)
{: .prompt-tip }

### Option page

From the **extension management page** in Firefox, users can also adjust several settings for the extension. For example, it is possible to disable specific rating sources or adjust the cache duration according to personal preference.  

![Extension Options](options.png)

## Future improvements  
Although many enhancements are possible, this was intended as a small project, so for now I decided to stop here and focus on new projects.  

Potential improvements include:  
- **Supporting other browsers**: Migrating the extension to **Manifest V3** would make it compatible with Chrome and other Chromium-based browsers, with only minor adjustments needed.  
- **Improving rating searches**: Incorporating additional information, such as the movie’s release year, could reduce confusion between films with the same title and increase the chances of finding accurate ratings.  
- **Adding new rating sources**: Expanding beyond IMDb, Rotten Tomatoes, and Allociné would provide an even broader perspective.  
- **Supporting other streaming platforms**: The existing backend mechanism could be adapted to work with services like Netflix, Disney+, and more, allowing the same extension to be used for a wider range of content.
