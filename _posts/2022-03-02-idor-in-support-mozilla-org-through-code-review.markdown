---
layout:	post
title:	"IDOR in support.mozilla.org through Code Review"
date:	2022-03-02
categories: [Hacking, Code Review]
image: /img/idor-in-support-mozilla-org-through-code-review/3.webp
tags: [Hacking, Code Review, Django, Web]
---

I was trying to improve my static analysis code, specifically django apps, so i decided to hack a random project in github. And i found kitsune. https://github.com/mozilla/kitsune

Kitsune is made by mozilla and according to them, it is what powers the support.mozilla.org

![](/img/idor-in-support-mozilla-org-through-code-review/1.webp)

So i downloaded it, and tried to hack it.

## FINDING THE IDOR

While going through all url endpoints, i found an interesting endpoint `url(r”^/(?P<question_id>\d+)/reply$”, views.reply, name=”questions.reply”)`

![](/img/idor-in-support-mozilla-org-through-code-review/2.webp)

It calls the function, views.reply. What makes this interesting is this part of the code

![](/img/idor-in-support-mozilla-org-through-code-review/3.webp)

Here, you can see that if you provide a `delete_images` post parameter, it will delete any image with the id you provided in the `delete_image` parameter with no checks if the user deleting the image is actually the owner of the image. Compare this to the real image delete function

![](/img/idor-in-support-mozilla-org-through-code-review/4.webp)

It has a proper authorization checks. Also, this functionality is not referenced anywhere in the front end since according to the mozilla team, the snippet is old.

So i reported it to mozilla bug bounty and asked their permission to actually try it in their staging server. And they agreed. After that, i was able to confirm the bug.

At the time of the report, support.mozilla.org is out of scope, but they still decided to reward me $1500 and added the domain in scope. You can read my whole report in <https://bugzilla.mozilla.org/show_bug.cgi?id=1754966>.

Thanks for reading, join the bounty hunter discord server: <https://discord.gg/bugbounty>