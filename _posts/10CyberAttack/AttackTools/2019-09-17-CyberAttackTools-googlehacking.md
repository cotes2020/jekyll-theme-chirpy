---
title: Meow's Testing Tools - googlehacking
date: 2019-09-17 11:11:11 -0400
categories: [10CyberAttack, CyberAttackTools]
tags: [CyberAttack, CyberAttackTools]
toc: true
image:
---

# googlehacking

[toc]

---

## Google Syntax Words

Google also allows keyword searches in specific parts of web pages using special syntax words. Additional commands, called special syntaxes, let Google users search specific parts This comes in handy when you'redealing with billions of web pages and need every opportunity to narrow your search results. Specifying thatyour query words must appear only in the title or URL of a returned web page is a great way to haveyour results get very specific without making your keywords themselves too specific.

`intitle`:
- `intitle`:
  - restricts your search to the titles of web pages. The variation,allintitle:
  - finds pages wherein all the words specified make up the title of theweb page. It's probably best to avoid the
- `allintitle`:
  - variation, because it doesn'tmix well with some of the other syntaxes.
> intitle:"george bush"
> allintitle:"money supply" economics

`inurl`:
- `inurl`:
  - restricts your search to the URLs of web pages. This syntax tends to work wellfor finding search and help pages, because they tend to be rather regular in composition.An allinurl:
  - variation finds all the words listed in a URL but doesn't mix well withsome other special syntaxes.

> inurl:help
> allinurl:search help

`intext`:
- `intext`:
  - searches only body text (i.e., ignores link text, URLs, and titles). There's an
- allintext:
  - variation, but again, this doesn't play well with others. While its uses arelimited, it's perfect for finding query words that might be too common in URLs or linktitles.
> intext:"yahoo.com"
> intext:html


`inanchor`:
- `inanchor`:
  - searches for text in a page's link anchors. A link anchor is the descriptivetext of a link.For example, the link anchor in the HTML code `<ahref="https://www.oreilly.com>O'Reilly and Associates</a>` is "O'Reilly and Associates."

> inanchor:"tom peters"

site:
- `site`:
  - allows you to narrow your search by either a site or a top-level domain.AltaVista, for example, has two syntaxes for this function (host:and domain:), butGoogle has only the one.

> site:loc.gov
> site:thomas.loc.gov
> site:edu
> site:nc.us

`link`:
- `link`:
  - returns a list of pages linking to the specified URL. Enter `link:www.google.com` and you'll be returned a list of pages that link to Google.You can includethe `https://` bit; you don't need it, and, indeed, Googleappears to ignore it even if you do put it in. `link:` works just as well with `"deep"URLs—https://www.raelity.org/apps/blosxom/` for instance—as with top-level URLs suchas raelity.org.

`cache`:
- `cache`:
  - finds a copy of the page that Google indexed even if that page is no longeravailable at its original URL or has since changed its content completely. This isparticularly useful for pages that change often.If Google returns a result that appears to have little to do with your query, you're almostsure to find what you're looking for in the latest cached version of the page at Google.
> cache:www.yahoo.com

`daterange`:
- `daterange`:
  - limits your search to a particular date or range of dates that a page wasindexed. It's important to note that the search is not limited to when a page was created,but when it was indexed by Google. So a page created on February 2 and not indexed byGoogle until April 11 could be found with daterange:
  - search on April 11.Remember also that Google reindexes pages. Whether the date range changes depends onwhether the page content changed. For example, Google indexes a page on June 1.Google reindexes the page on August 13, but the page content hasn't changed. The datefor the purpose of searching with daterange:
  - is still June 1.Note that daterange:
  - works with Julian, not Gregorian dates (thecalendar we use every day.) There are Gregorian/Julian converters online, but if you wantto search Google without all that nonsense, use the FaganFinder Google interface`(https://www.faganfinder.com/engines/google.shtml)`, offering daterange:searchingvia a Gregorian date pull-down menu. Some of the hacks deal with daterange:searching without headaches, so you'll see this popping up again and again in the book.
> "George Bush" daterange:2452389-2452389
> neurosurgery daterange:2452389-2452389


`filetype`:
- `filetype`:
  - searches the suffixes or filename extensions. These are usually, but notnecessarily, different file types. I like to make this distinction, because searching forfiletype:htm and `filetype:html` will give you different result counts, eventhough they're the same file type. You can even search for different page generators, suchas ASP, PHP, CGI, and so forth—presuming the site isn't hiding them behind redirectionand proxying. Google indexes several different Microsoft formats, including:PowerPoint(PPT), Excel (XLS), and Word (DOC).
> homeschooling filetype:pdf
> leading economic indicators"  filetype:ppt


`related`:
- `related`: finds pages that are related to the specified page. Notall pages are related to other pages. This is a good way to find categories of pages; asearch for related:
- google.com would return a variety of search engines,including HotBot, Yahoo!, and Northern Light.
> related:www.yahoo.com
> related:www.cnn.com


`info`:
- `info`:
  - provides a page of links to more information about a specified URL. Informationincludes a link to the URL's cache, a list of pages that link to that URL, pages that arerelated to that URL, and pages that contain that URL. Note that this information isdependent on whether Google has indexed that URL or not. If Google hasn't indexed thatURL, information will obviously be more limited.
> info:www.oreilly.com
> info:www.nytimes.com/technology


`phonebook`:
- `phonebook`: looks up phone numbers.phonebook:
> John Doe CAphonebook:(510) 555-1212
