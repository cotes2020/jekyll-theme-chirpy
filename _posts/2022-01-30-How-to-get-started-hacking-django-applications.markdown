---
layout:	post
title:	"How To Get Started Hacking Django Based Applications"
date:	2022-01-30
medium_url: https://systemweakness.com/how-to-get-started-hacking-django-applications-f407564df9c7
categories: [Hacking, Code Review]
tags: [Hacking, Django, Code Review, Web]
---

Django is a python based web framework. In this writeup, i will teach you how to analyze django based applications . For this writeup, i will be using wagtail for examples. When doing static code analysis, for me , the most important part is identifying how routes are registered and handled, that is what we will cover in this writeup.

Before we start, i recommend you having visual studio code, since it is lightweight and its easy to analyze code using it, you can just use ctrl+click on functions to follow their definitions which helps alot. But, you can still use an editor of your choice if you want.

### Url Dispatcher

All django applications has a file called urls.py. This file contains all the url endpoint available in the application. There are 3 function that allows us to register our endpoints, path, `re_path`, and url. path is the main way to register endpoints. `re_path` is like path but support regex. url is just a reskin of `re_path`. Lets take a look at wagtail for example.

![](/img/1*lPimpxKiWKeBRFuMRFq5Dw.png)Here, you can see that 4 endpoints are registered using the path function. The second argument is the callback function that will be called when that endpoint is accessed. In line 17, when we access the search/ endpoint, it will call the `search_views.search` function. The name argument is not that important. Alternatively, we can also provide another list of endpoints in the second argument, like what is done in line 12, 14, and 15. In line 14 for example, it includes `wagtailadmin_urls`

![](/img/1*BkgKfLlD_NX270VYZQm28A.png)Now if we access for example `admin/account/` it will call the `account.account` function as seen in line 59.

There are times where we need certain parts of the url as a variable, just like in line 39, in this case, the variables will be passed as an argument to the function to be called. If you follow the `bulk_actions` function, you will see this.

![](/img/1*Cr3ZVtPsjvFsZEHIqTzykA.png)The first argument is always the request, and the second, third, and fourth argument is the variable from the url path.

### Class Based Views

This is by far the most confusing part for me. But, i will try to make it as simple as possible.

There are times where you see this in the callback.

![](/img/1*eFGOn4zmc77jsSufUccuHQ.png)This means, that it is using a class based view. In the django, there has a class called View. According to the django documentation, this is the flowchart of the View class

![](/img/1*ccyvYknVLrmWe0aFbrKd9A.png)

The setup is the setup, not that important for us, the dispatch, is the function that actually handles the request. By default, it checks the method of the request, check if it is in the `http_method_names` variable, and call the associated function to the method. For example, if the http method is post, it will call the post function. The `http_method_not_allowed` is self explanatory, and the options function handle the request if the http method is option. This View class is supposed to be inherited by other class.

Lets take the `LoginView` from wagtail as an example, on its roots, `LoginView` inherits the class `ProcessFormView`, which inherits the View class.

![](/img/1*mJXUv-LHWhwVAEDYUtPQZg.png)So when the `LoginView.as_view` function is called, the dispatch function from the View class will be called. It will check the http method and execute the corresponding function depending on the http method.

The class `LoginView` inherits this ProcessFormView. And you can see in `LoginView` the `form_valid` function, which handles the authentication. This `form_valid` function is called from the ProcessFormView class from above in the post function.

![](/img/1*KD_UE1Y01FB4Z0UDT-8SHg.png)Lets take another View class as an example.

![](/img/1*h9ml69NxLnBAw4d1faM0Fg.png)Here, it uses the class TemplateView.

![](/img/1*Nc7V2TaRl-dU-LYfTiOxXA.png)Like before, when the `as_view` function is called, it will call the dispatch function, which check if the http method is get, if it is, call the get function. It also inherits two more class, TemplateResponseMixin and ContextMixin. This provide the required functionalities in the get function. In the get function, you can see that it calls the `render_to_response` function which is a function in the `ContextMixin` class. What it does is render the template.

![](/img/1*JzPGVDuP4fcPkvkNKP-Pxw.png)The `get_template_names` just return the `template_name` variable.

![](/img/1*q2J7O8B7ueQsHhBwI4PREA.png)As you remember, we set the value of this `template_name` when routing the endpoint.

![](/img/1*nqFMV3Xlmk1Vf4rPJy7Hkg.png)So to summarize, when the `TemplateView.as_view` is called, it will call the dispatch function, which checks if the http method is get, if it is it will call the get function in the TemplateView class, the get method, calls the `render_to_response` which render the template we provided in the `template_name`.

Class based view is really confusing so i’ll add another example, you can skip to the next part if you want.

![](/img/1*uWj7-qvtbXOh1xI1TPViag.png)Here, you can see that it uses the class `WorkflowAction`. `WorkflowAction` inherits `BaseWorkflowFormView` which inherits the View class.

![](/img/1*QnK92hbdF4KhjSoaJX1jpA.png)In the `BaseWorkflowFormView`, you can see that it overrides the dispatch function.

![](/img/1*mGNyne9ZOgadXT3X-EZDbw.png)I wont go to what it does but in the end of the dispatch, you can see that it calls the original dispatch method of the super class, which is the View class to continue the flow of the original View Class. Like before, the dispatch method in the View class checks the http method, and call the corresponding function. You can see in `BaseWorkflowFormView` that it has both get and post function,

![](/img/1*6SBymaiSkmaDleIiDIQ-Ow.png)but the post function is overridden by WorkflowAction when that class inherits `BaseWorkflowFormView` so this post method is called instead when the http method is post.

![](/img/1*iv0VFMY5K4F8cTd-uxP_kw.png)
### Templates

In django, pages are often times rendered using templates. We will use the tags/ as an example.

![](/img/1*nm50BSGW3bEx6mPGkAzXLg.png)

Here, it uses the class TemplateTagIndexView

![](/img/1*IZwYVlut3BE481a2h5G6Uw.png)

Which inherits BaseAdminDocsView

![](/img/1*XR_qQFpoOuOXzz_3CgW75A.png)

which inherits TemplateView. We already talked about TemplateView above. So, we know that TemplateView has a get function, which renders the template with a context provided by `get_context_data`.

![](/img/1*A31d2GO0nvVnzyzvaKx4GA.png)

From the screenshots above, you can see that the `template_name` variable is overridden by the TemplateTagIndexView class. Now you can see that there is a parameter in `render_to_response` called context, the context is like a variables used by the template when rendering. These context are just dictionaries.

The TemplateTagIndexView class override the `get_context_data` function.

![](/img/1*GROthfvTktRxM6EMApryog.png)

You can see that it makes a new dictionary called tags, and add it to the original `get_context_data`. This variables are then used when rendering the template.

![](/img/1*xc-cfrDtvh5263Jo6XkxhA.png)
### End

This is the end of the writeup, by now, you should know how a django application operates and how to test it. Normally, i made these writeups as a reference for myself when i want to tackle django hacking again, but i decided to publish it. Thanks for reading.

Join the bounty hunter discord server: <https://discord.gg/bugbounty>

  