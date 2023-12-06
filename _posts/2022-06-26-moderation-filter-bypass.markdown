---
layout:	post
title:	"Moderation Filter Bypass in support.mozilla.org"
date:	2022-06-26
categories: [Hacking, Bug Bounty]
image: /img/moderate1.png
tags: [Hacking, Code Review, Django, Web, Bug Bounty]
---

# SUMMARY
Recently, the triager at mozilla pointed out to me, that when replying a link to a question, the reply will be mark as spam, and will be up for moderation.   
![](/img/moderate.png)    
So i decided, to attack this functionality and try to bypass it

# CODE REVIEW
When making a reply, we use the `AnswerForm` form
```python
def reply(request, question_id):
    """Post a new answer to a question."""
    question = get_object_or_404(Question, pk=question_id, is_spam=False)
    answer_preview = None

    if not question.allows_new_answer(request.user):
        raise PermissionDenied

    form = AnswerForm(request.POST, **{"user": request.user, "question": question}) # LOOK HERE
```
This `AnswerForm` class has a method call `clean`. This is the function responsible for checking if the reply is a spam or not.
```python
class AnswerForm(KitsuneBaseForumForm):
    """Form for replying to a question."""

    content = forms.CharField(
        label=_lazy("Content:"),
        min_length=5,
        max_length=10000,
        widget=forms.Textarea(attrs={"placeholder": REPLY_PLACEHOLDER}),
    )

    class Meta:
        model = Answer
        fields = ("content",)

    def clean(self, *args, **kwargs):
        """Override clean method to exempt question owner from spam filtering."""
        cdata = super(AnswerForm, self).clean(*args, **kwargs) # LOOK HERE
        # if there is a reply from the owner, remove the spam flag
        if self.user and self.question and self.user == self.question.creator:
            cdata.pop("is_spam", None)

        return cdata
```
Here, you can see that the clean, simply call the clean function of the parent class which is the `KitsuneBaseForumForm`. This is the `KitsuneBaseForumForm` class. I removed a few parts of it which is unimportant
```python
class KitsuneBaseForumForm(forms.Form):
   def __init__(self, *args, **kwargs):
        #UNIMPORTANT SNIPPET

    def clean(self, *args, **kwargs):
        cdata = self.cleaned_data.get("content")
        
        # UNIMPORTANT SNIPPET

        if not (
            self.user.groups.filter(name__in=TRUSTED_GROUPS).exists()
            or self.user.has_perm("flagit.can_moderate")
            or self.user.has_perm("sumo.bypass_ratelimit")
        ) and check_for_spam_content(cdata): # LOOK HERE
            self.cleaned_data.update({"is_spam": True})

        return self.cleaned_data
```
Here, you can see that it calls the `check_for_spam_content` on our user input. If it returns true, our reply will be mark as spam and will be up for moderation.
```python
def check_for_spam_content(data):
    digits = "".join(filter(type(data).isdigit, data))
    is_toll_free = settings.TOLL_FREE_REGEX.match(digits)

    is_nanp_number = match_regex_with_timeout(settings.NANP_REGEX, data)

    has_links = has_blocked_link(data) # INTERESTING PART

    return is_toll_free or is_nanp_number or has_links
```
In the `check_for_spam_content` function, you can see the function that checks our input for any links.`has_blocked_link`
```python
def has_blocked_link(data):
    for match in POTENTIAL_LINK_REGEX.finditer(data):
        tld = match.group(1).upper()
        if tld in VALID_TLDS: # VULNERABLE PART OF THE CODE
            full_domain = match.group(0).lower()
            in_allowlist = False
            for allowed_domain in settings.ALLOW_LINKS_FROM:
                split = full_domain.rsplit(allowed_domain, 1)
                if len(split) != 2 or split[-1]:
                    continue
                if not split[0] or split[0][-1] == ".":
                    in_allowlist = True
                    break
            if not in_allowlist:
                return True
    # UNIMPORTANT SNIPPET
    return False
```
By default, it returns False, which is what we want. You can see that before anything else, it first checks if the tld of our input is in the `VALID_TLDS`. Lets see what this `VALID_TLDS` is
```python
# downloaded from https://data.iana.org/TLD/tlds-alpha-by-domain.txt
path = os.path.join(os.path.dirname(__file__), "tlds-alpha-by-domain.txt")

with open(path) as f:
    VALID_TLDS = set(f.read().splitlines()[1:])
```
It is the `tlds-alpha-by-domain.txt` list from iana.org. Opening this txt file, i found out that they are using an outdated version of the list.
```
# Version 2020062200, Last Updated Mon Jun 22 07:07:01 2020 UTC
AAA
AARP
ABARTH
...
```
So i thought, what if there are new tlds that are registered between 2020 and today. So, i downloaded the newest tld list from <https://data.iana.org/TLD/tlds-alpha-by-domain.txt>, compare the two and get all the tld that doesnt exist on both txt files. 
```console
$ cat tlds-alpha-by-domain.txt* | sort | uniq -u
AFAMILYCOMPANY
AIGO
BUDAPEST
CASEIH
CEB
CSC
DUCK
FUJIXEROX
GLADE
INTEL
IVECO
JCP
LIXIL
LUPIN
METLIFE
MUSIC
NATIONWIDE
NEWHOLLAND
OFF
ONYOURSIDE
QVC
RAID
RIGHTATHOME
RMIT
SCJOHNSON
SHRIRAM
SPA
SPREADBETTING
SWIFTCOVER
SYMANTEC
# Version 2020062200, Last Updated Mon Jun 22 07:07:01 2020 UTC
# Version 2022040300, Last Updated Sun Apr  3 07:07:01 2022 UTC
XN--3OQ18VL8PN36A
XN--4DBRK0CE
XN--KPU716F
XN--PBT977C
```
And write a simple python script to see which of these tlds, doesnt belong in the old tld list of mozilla.
```python
import os
 
#LIST OF TLDS THAT IS NOT ON BOTH TEXT FILES
lamaw = ["AFAMILYCOMPANY","AIGO","BUDAPEST","CASEIH","CEB","CSC","DUCK","FUJIXEROX","GLADE","INTEL","IVECO","JCP","LIXIL","LUPIN","METLIFE","MUSIC","NATIONWIDE","NEWHOLLAND","OFF","ONYOURSIDE","QVC","RAID","RIGHTATHOME","RMIT","SCJOHNSON","SHRIRAM","SPA","SPREADBETTING","SWIFTCOVER","SYMANTEC","XN--3OQ18VL8PN36A","XN--4DBRK0CE","XN--KPU716F","XN--PBT977C"] 
 
path = os.path.join(os.path.dirname(__file__), "tlds-alpha-by-domain.txt")
 
for i in lamaw:
    with open(path) as f:
        VALID_TLDS = set(f.read().splitlines()[1:])
        for x in VALID_TLDS:
            if x==i:
                continue
    print(i)
```
Running it, i found all the valid tlds, that are not on the list of mozilla.   
![](/img/shish.png)    
Now, we can make a comment with a url with any of these tld, like `https://evil.spa` and it will not be marked as spam, bypassing the filter. 
![](/img/poc.png)      

Unfortunately, the only thing we can achieve with this bug is spam bypass which is out of scope by mozilla. If this was in any other functionalities, this would have been a cool bug.    
![](/img/moderate1.png)     
Thank you for reading.
