---
layout:	post
title:	"Unauthenticated Stored XSS on Django-Markdownx"
date:	2022-03-02
categories: [Hacking, Code Review]
image: /img/markdownx-xss-popup.PNG
tags: [Hacking, Code Review, Django, Web]
---


# INTRODUCTION
Django-markdownx is a famous markdown library for python. According to github, it is used by 1.6k projects, on githubs, that doesnt include closed source projects and websites.  
![](/img/Markdownx_Used_by.png)   
In this writeup, i will show you a bug that i find in this django plugin

# CODE REVIEW
Starting from the `urls.py`, it only has two endpoints   
```py
urlpatterns = [
    url('upload/', ImageUploadView.as_view(), name='markdownx_upload'),
    url('markdownify/', MarkdownifyView.as_view(), name='markdownx_markdownify'),
]
```     
The upload endpoint is pointing to the ImageUploadView class.    
```py
class ImageUploadView(BaseFormView):
    form_class = ImageForm
    success_url = '/'

    def form_invalid(self, form):
        #Uninteresting snippet

    def form_valid(self, form):
        response = super(ImageUploadView, self).form_valid(form)

        if self.request.headers.get('x-requested-with') == 'XMLHttpRequest':
            image_path = form.save(commit=True)
            image_code = '![]({})'.format(image_path)
            return JsonResponse({'image_code': image_code})

        return response

```    
It is an BaseFormView class which is just the FormView Class. You can see how the FormView class works in diagram here <https://www.brennantymrak.com/articles/formviewdiagram>. It uses the class ImageForm for the form and it save the form. Here is the ImageForm class        
```py
class ImageForm(forms.Form):
    image = forms.FileField()
    
    _SVG_TYPE = 'image/svg+xml'

    def save(self, commit=True):
        image = self.files.get('image')
        content_type = image.content_type
        file_name = image.name
        image_extension = content_type.split('/')[-1].upper()
        image_size = image.size

        #Unimportant code

        if (content_type.lower() == self._SVG_TYPE
                and MARKDOWNX_SVG_JAVASCRIPT_PROTECTION
                and xml_has_javascript(uploaded_image.read())):

            raise MarkdownxImageUploadError(
                'Failed security monitoring: SVG file contains JavaScript.'
            )

        return self._save(uploaded_image, file_name, commit)
```
If the file is svg, it checks it using the `xml_has_javascript()` function.     
```py
def xml_has_javascript(data):
    from re import search, IGNORECASE, MULTILINE

    data = str(data, encoding='UTF-8')
    pattern = r'(<\s*\bscript\b.*>.*)|(.*\bif\b\s*\(.?={2,3}.*\))|(.*\bfor\b\s*\(.*\))'

    found = search(
        pattern=pattern,
        string=data,
        flags=IGNORECASE | MULTILINE
    )

    if found is not None:
        return True
    #Unimportant Code
    return False

```    
Here, it only checks if it has a `\<script>` or `<if>` or `<for>` node. However, there are ways to achieve xss without using any of those, i made a poc with it posted in github <https://gist.github.com/noobexploiterhuntrdev/c4db7e87841f43f3befdeb1de4f18092>.    

# REPRODUCTION
I dont code myself, hopefully tho, i found a github project using this library. So i installed it and hosted it in my vps. <https://github.com/vladyslavnUA/foodanic>.

I made the file upload request to `upload/`, with the svg payload that i have.    
![](/img/Markdownx-File-Upload.png)    
The filename and the directory of the uploaded image, is shown in the response. Visiting this endpoint, will give us the xss popup that we expected    
![](/img/markdownx-xss-popup.PNG)     
Our xss payload worked    

# CLOSING NOTES
I tried my best to disclose this to the maintainer of the project, but they are really unresponsive. I reported it last febuary, and it is now April, so i decided to disclose it. Thanks for reading