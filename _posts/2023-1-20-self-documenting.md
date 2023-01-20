---
layout: post
title: 2023.01.20
categories: "Interesting Articles"
tags: 2023 productivity reading brain loneliness aging stress diversity equity inclusion poor classism "software engineering" versioning semver "guard clauses" exceptions validation ardalis infosec "tanya janca" "adam shostack" threats "personal development" learning teams feedback
---

You don't have to read it, but you just might learn something.

<!--more-->

## Leading Thought

!["Caring for myself is not self-indulgence, it is self-preservation, and that is an act of political warfare. ~Audre Lorde"](https://www.azquotes.com/picture-quotes/quote-caring-for-myself-is-not-self-indulgence-it-is-self-preservation-and-that-is-an-act-audre-lorde-45-67-08.jpg){: width="500" }

## Prime

### [What does reading on screens do to our brains?](https://www.bbc.co.uk/ideas/videos/what-does-reading-on-screens-do-to-our-brains/p09xx6qw)

Interesting (and short) video about the what happens to your brain when you read and the benefits of literacy. Interestingly, the medium plays a large role not only in *how* we read, but the effects on comprehension, empathy, and analysis. Unsurprisingly, digital lends itself more to short bursts of reading where parts may be skimmed -- think news or blogs. Want to read something cognitively or emotionally challenging? Leave that to good old dead-tree editions.

### [the diminishing returns of productivity culture ](https://annehelen.substack.com/p/the-diminishing-returns-of-productivity)

Interesting post here about how the promise of technology to give everyone more time for leisure has done the opposite and increased the amount of work we do, sometimes with more responsibilities than were once expected, for the same amount of pay. In other cases, technology has taken skilled trades like machinists, removed the *skill*
 in many ways, and increased the expected output.

 As a technology worker, it can be hard to balance the thrill of creating something that makes the monotonous easier for someone so that they are able to do something *better* with their time, against the empathy to understand what the real impact may be. Definitely worth a read if for no other reason than to understand how expectations to be *productive* have changed and the impact this has on how we live.

> Knowledge workers have embraced our own deskilling, and the subsequent de-valuing of our labor, and rebranded it as personal productivity.

### [My Mom Has No Friends The loneliness of growing older made bearable with playdates.](https://www.thecut.com/2022/08/helping-my-mom-make-friends.html)

This is a great reminder that loneliness is both an epidemic and a killer. It's interesting how things Im reading seem to collide unexpectedly; this subject came up in a book Im listening to, **Burnout: The Secret to Unlocking the Stress Cycle** by Emily Nagoski PhD and Amelia Nagoski DMA (while this book is targeted toward women, I highly recommend that men read it to gain more insight and empathy into the stressors women deal with).

While this is a good story about the author and her mom, there are some important lessons in here:

- as you get older, you lose friends for many reasons and it becomes much harder to make new friends
- the stress of caring for a parent and children can be overwhelming
- take time to know your parents, assuming you don't have a toxic relationship; they won't be around forever

> Social isolation, for people over 50, is correlated with a 50 percent increased risk of dementia, according to the CDC, and a 32 percent increased risk of stroke. Loneliness is also associated with higher rates of anxiety, depression, and suicide. Prolonged isolation is a health risk on the level of smoking. (And older members of marginalized communities are at an even higher risk for all of the above when they’re socially secluded.)

[Return to Top](#leading-thought)

## DE&I

### [Poor teeth](https://aeon.co/essays/there-is-no-shame-worse-than-poor-teeth-in-a-rich-world)

I stumbled across this story from 2014 which should be a must-read for anyone who has been lucky enough to grow up with dental insurance and the money to maintain good oral health. While the focus is on how we judge people when they have bad teeth, the real theme is how we judge each other in myriad ways, and look for ways to feel superior to others. The author notes the popularity of things like the blog *People of Walmart* to demonstrate how the poor are ridiculed and subject to bigotry because they don't have the privilege of better clothes, food, or healthcare. There are a lot of people who simply don't understand how expensive being poor is, nor the lifestyle decisions required to survive.

It's easy to fall into this trap when you don't know any better. Read the article; it just may make you take a fresh look at the people around you and consider the challenges they may be facing. While it may cost a bit more in taxes to provide universal healthcare to everyone, I expect the initial cost would go down as people become less dependent on emergency services to treat problems, and can manage and prevent problems before they occur. For the vast majority of us, where a job-loss away from losing access to preventive care and becoming a member of the vilified.

> Poor teeth, I knew, beget not just shame but more poorness: people with bad teeth have a harder time getting jobs and other opportunities. People without jobs are poor. Poor people can’t access dentistry – and so goes the cycle.

[Return to Top](#leading-thought)

## Engineering

### [Major Version Numbers are Not Sacred](https://tom.preston-werner.com/2022/05/23/major-version-numbers-are-not-sacred.html)

Assuming you use any sort of libraries in your projects, you're probably exposed to *semantic versioning* (SemVer) almost every day. Chances are good, too, that you've run into a breaking change in something that didn't have a major revision. [SemVer is pretty well-defined in how it's meant to work](https://semver.org), so *why* does this happen? Chances are it has something to do with marketing and the practice of major revisions corresponding to big product releases.

We need to get away from this practice because a breaking change in SemVer *requires* a major revision change -- it's part of the common understanding. Not following the standard for *reasons* breaks the contract and frustrates consumers. This post from [Tom Preston-Werner](https://github.com/mojombo/), creator of SemVer, does a great job of explaining the why's and how's of SemVer, as well as solutions for decoupling marketing from versioning. Definitely worth a read.

> I want to live in a world where every breaking change comes gift-wrapped in a major release. I want the promise of SemVer to be fully realized. I think we can get there by rejecting the tyranny of sacred major version numbers. If you feel the same, I hope you’ll join me in embracing this philosophy.

### [Guard Clauses and Exceptions or Validation?](https://ardalis.com/guard-clauses-and-exceptions-or-validation/)

[Steve *Ardalis* Smith](https://twitter.com/ardalis?s=20&t=yo5_Mk3F6pl5_v3uyJpXXA) has always got a great take on things and this look at handling bad data is no exception. While it has a focus on C#, the principles can be applied universally.

[Return to Top](#leading-thought)

## Infosec

### [Threats: What Every Engineer Should Learn From Star Wars](https://threatsbook.com)

This looks like it has the potential to be a great book from [Adam Shostack](https://shostack.org/about/adam), and the fact that it's got the attention of [Tanya Janca](https://shehackspurple.ca) really makes me want to pick it up. Release date is February 7, but you can pre-order it now.

[Return to Top](#leading-thought)

## Personal Development

### [Variability, Not Repetition, is the Key to Mastery](https://www.scotthyoung.com/blog/2022/10/26/variable-mastery/)

Interesting read about how learning works. Understanding and using alternate ways to solve a problem helps to master them -- think about something you do well and then try to think about all they ways you may know to accomplish that thing. Thought provoking, at the least.

Some of the best programmers I know work in multiple languages and styles (e.g., Object Oriented *and* Functional Programming,or C# and Go). This seems to align with the ideas here. Others I know work coding [katas](https://en.wikipedia.org/wiki/Kata) regularly to improve their speed at solving a problem; I'd be curious to know haw many of them are changing languages or styles regularly and benefitting from the ideas here.

If you are trying to learn something new, some variability may help, but start slow and learn the basics. As you work toward mastery, look for new ways to do it and mix up your practice. I'm definitely going to take a look at some of the things I do to see if there are alternate approaches that may help me get better. It can't hurt, right?

[Return to Top](#leading-thought)

## Teams

### [How to Encourage Your Team to Give You Honest Feedback](https://hbr.org/2022/10/how-to-encourage-your-team-to-give-you-honest-feedback)

If you lead people formally, you may really want feedback from your direct reports, but how do you convince people whose fate is in your hands to be honest with you to help you grow? How about convincing those who are part of a culture that sees this as taboo? Some good stuff here that smacks of **Radical Candor** with a bonus nod to the *[Conscious competence theory of learning a new skill](https://en.wikipedia.org/wiki/Four_stages_of_competence)*.

[Return to Top](#leading-thought)
