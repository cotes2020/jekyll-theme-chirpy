---
layout: post
title: 2023-02-17
permalink: /posts/:title/2023-02-17
date: 2023-2-17 00:01 -0500
categories: "Interesting Articles"
tags: 2023 groupthink management bus-to-abilene pull-requests code-reviews work-from-home pomodoro adhd introverts brainstorming ai facial-recognition institutional-racism bias engineering entity-framework database-first orm javascript script uber testing ephemeral-environment slate logging asp-net-core 
---

You don't have to read it, but you just might learn something.

<!--more-->

## Leading Thought

![Twitter post from "Agile Otter" Tim Ottinger (@tottinge): What’s worse than demotivated people? People highly motivated to do the wrong things.](../../../assets/img/self-documenting/otter-wrong-things.png){: width="500" height="140" .left-no-float }

---

## Prime

### [The Abilene Paradox: The Management of Agreement](http://web.mit.edu/curhan/www/docs/Articles/15341_Readings/Group_Dynamics/Harvey_Abilene_Paradox.pdf)

I just finished listening to the book [Quiet: The Power of Introverts in a World That Can't Stop Talking](https://www.powells.com/book/quiet-9780307352156) and there was mention of a phrase common in the US Army: "Are we on the bus to Abilene?". I'm not sure of how common this is in the Army but I did find reference to General Colin Powell talking about it, so it may be the case. Regardless of the truth, I went in search of the original article by management expert Jerry B. Harvey, published in  1974.

This is one of those things that I stumble across and think *everyone* should read, but especially senior leaders of an organization. The core idea is that bad decisions are often made because people are unwilling to rock the boat given any number of fears and, instead, go along to get along much to the detriment of the organization. Harvey starts out with a personal experience -- the origin of the term *The Bus to Abilene* -- and uses a couple of real-world examples, including the Watergate scandal to demonstrate how *The Abilene Paradox* works. It's not the shortest read, but it's absolutely worth your time to give it a look.

Next time you're part of a group doing something no one seems to be happy doing, ask the question: **Are we on the bus to Abilene?**

> A well-known example of such faulty conceptualization comes to mind. It involves the heroic sheriff in the classic Western movies who stands alone in the jailhouse door and singlehandedly protects a suspected (and usually innocent) horse thief or murderer from the irrational, tyrannical forces of group behavior—that is, an armed lynch mob. Generally, as a part of the ritual, he threatens to blow off the head of anyone who takes a step toward the door. Few ever take the challenge, and the reason is not the sheriff’s six-shooter. What good would one pistol be against an armed mob of several hundred people who really want to hang somebody? Thus, the gun in fact serves as a face-saving measure for people who don’t wish to participate in a hanging anyway. (“We had to back off. The sheriff threatened to blow our heads off.”)

### [PRs: Shift Left, Please (Part One)](https://medium.com/pragmatic-programmers/prs-shift-left-please-part-one-b0f8bb79ef2b) & [(Part Two)](https://medium.com/pragmatic-programmers/prs-shift-left-please-part-2-536b48be2476)

One of the people that I learned good code reviewing from was [Brian Friesen](https://www.linkedin.com/in/randomskunk/) when I was lucky enough to be on the same team with him (one of the things he would say was, "I like what you did here, but have you thought about this?"). But, more so, I learned to actually dig in and give time to the process; to me, this not only shows respect for the engineer who's code is being reviewed, but for whomever is funding the work.

That said, I would definitely include myself in the group of developers who think that Pull Requests (like unit tests) create a false sense of security. Far too often, developers are busy solving their own issues, or may have a personality that is averse to conflict leading to the all-too-common 'Looks good' approval (a small *bus to Abilene*). To be fair, conflict avoidance can exist while pairing or mobbing but, hopefully, over time, people begin to build relationships leading to trust, which leads to better communication and solutions. A response I saw to the article pointed out that in a company, you are "...applying a low trust process that makes sense in open source contributions to a setting where you know your team."

Whether you believe PRs are effective or not, this pair of posts is definitely worth a read.

> In a written forum, however, a comment interpreted as offensive (whether or not the reviewer thought it was) can burn for a while if not properly clarified. Sometimes the reviewer ends up offended that the person who wrote the code was offended, in which case both parties can feel less safe in the team. This decrease in the team’s collective trust level can reduce its effectiveness. As a result, we learn to spend extra time carefully wording a question or comment, which in turn increases inefficiency in and distaste for the PR process.

### [How to minimize distractions when you work from home](https://www.theverge.com/23274524/work-from-home-distractions-wfh-how-to)

Some solid ideas here for limiting distractions and becoming more productive when working from home.  While I've tried the [Pomodoro Technique](https://todoist.com/productivity-methods/pomodoro-technique) and am curious about the modified *52/17* version. If you work from home or have ADHD, you may find something useful here to try.

### [Science shows brainstorms don’t work. Why do we still use them?](http://openforideas.org/blog/2016/11/23/science-shows-brainstorms-dont-work-why-do-we-still-use-them/)

A second intersection from the book [Quiet: The Power of Introverts in a World That Can't Stop Talking](https://www.powells.com/book/quiet-9780307352156), the effectiveness of brainstorming (or lack thereof) was mentioned in the context of the negative impact on introverts in the workplace. It's probably not coincidence that this post managed to find it's way into my sources, but this makes the same point: we know that brainstorming doesn't work the way we expect it to, yet we continue to use it anyway.

Definitely worth a read if you are still using brainstorming to generate ideas. The post includes some alternate strategies that may not only help you get the results you want, but would likely have a side-effect of giving introverts a louder voice.

> But group creativity isn’t a bad thing. Your staff are oozing with knowledge, experience and potential. And there are really great ways of harnessing that and using it to generate effective ideas. If we can simply move beyond brainstorms.

[Return to Top](#leading-thought)

---

## Coming Soon

### [Hacking With The Homies Developer Conference](https://www.hackingwiththehomies.org/) 
(Feb 23 - Feb 25, 2023 | Detroit, MI & Virtual)

Hosted by Detroit Black Tech, this conference is focused on helping Black and Brown Software Developers level up in their careers and helping companies connect to a more diverse segment of the working population.

With a virtual attendance cost of just $60, and an in-person, early bird price of $90, this is definitely a conference worth checking out.

### [axe-con 2023](https://www.hackingwiththehomies.org/) 
(Mar 15 - Mar 16, 2023 |  Virtual)

If Digital Accessibility (A11y) is your thing, then this conference is for you -- and it's FREE! I've attended in the past and there are always great talks; this year looks like no exception. Included in the speakers is [Imani Barbarin](https://twitter.com/Imani_Barbarin) whom I've followed on Twitter for quite a while. She runs [crutchesandspice.com](https://crutchesandspice.com/) and always has honest thoughts on both the abled and disabled communities.

[Return to Top](#leading-thought)

---

## AI

### [Facial recognition tool led to mistaken arrest, lawyer says](https://apnews.com/article/technology-louisiana-baton-rouge-new-orleans-crime-50e1ea591aed6cf14d248096958dccc4)

Another example here about the danger posed by AI trained on either biased or faulty data. In this case, the AI misidentified a man for a crime in Louisiana, even though he had never been to Louisiana. The fact that there hasn't been Federal legislation to ban the use of AI systems until an effective method of removing bias is found is alarming, to say the least. Far too many cases exist in systems of all sorts, from sentencing recommendations to flagging people for unemployment fraud, without a truly effective way to understand how the AI arrived at a decision. At a minimum this can cause victims minor pain or inconvenience; at it's worst, people face financial ruin or wrongful detention. We can do better. We *need* to do better.

> Facial recognition systems have faced criticism because of their mass surveillance capabilities, which raise privacy concerns, and because some studies have shown that the technology is far more likely to misidentify Black and other people of color than white people, which has resulted in mistaken arrests.

### [The EU wants to put companies on the hook for harmful AI](https://www.technologyreview.com/2022/10/01/1060539/eu-tech-policy-harmful-ai-liability/)

Understanding the problems with AI technology, the EU is leading the way on making it possible for people to hold companies liable for the harms done by AI systems. While the law would make it possible for class action suits, the onus is still on the complainants to prove that the system caused harm, a herculean task.

Definitely a good start at addressing some of the problems, it will be interesting to see how the US follows suit. Given the issues seen already, from misidentified unemployment fraud, to misidentified suspects in crimes, to unfair sentencing, we need to make sure that algorithms are solid, provable, and based on unbiased data. If financial and criminal penalties put a chill on the development of AI systems, then this may not be the worst thing if it prevents the unleashing of faulty systems on the public.

---

## Engineering

### [An amusing story about a practical use of the null garbage collector](https://devblogs.microsoft.com/oldnewthing/20180228-00/?p=98125)

As engineers, we generally like hard problems. Edge cases and efficiency are things many developers obsess about, but do we always put enough thought into true mitigation or when we can ignore a known issue entirely? This is a great anecdote to file away for when you find yourself struggling to determine whether the fix is worth the effort, or if a more abrupt solution is acceptable (*Have you tried turning it off and on again?*).

### [Example of an Entity Framework Core Database First development workflow in a team environment](https://davecallan.com/entity-framework-core-database-first-development-workflow-in-a-team-environment/)

If you do database-first development using EF Core, or think you want to try it, this may be something you want to take a look at. The author has a well-thought out approach, with great details in the presentation. 

Even if you think code-first is the only way you want to develop, it never hurts to see how someone else approaches a problem. It may be worth some experimenting with this flow to see if your opinion shifts at all.

### [Difference Between Async & Defer Attributes in JavaScript](https://dev.to/catherineisonline/difference-between-async-defer-attributes-in-javascript-57ff)

Good, clear post here explaining what the *defer* and *async* attributes do when applied to to ```script``` tags in a web page. If you've been around awhile, you probably have tucked away the strategy of placing script tags at the end of a page in order to allow HTML to load first. With the *defer* attribute, you get the same effect while keeping your scripts in the HEAD of the document; *Async* allows scripts to download  in in parallel, but loses any guarantee of completion order.

---

## Testing

### [Simplifying Developer Testing Through SLATE](https://www.uber.com/en-IN/blog/simplifying-developer-testing-through-slate/)

Have to thank [Carter Wickstrom](https://www.linkedin.com/in/carterwickstrom/) for boosting this on LinkedIn; it's a fascinating post about how Uber is using tenancy and ephemeral environments in conjunction with their Production environment to test new code.

I think a lot of people would agree that, in an ideal world, observability was at a place where we could truly push code to Prod and understand immediately if something had broken and rollback automatically. This would, in theory, remove the need for testing environments. Until we get to that point, this is a really interesting approach. Like everything, of course, most companies probably don't need something of this magnitude; however, if your environment is a complex web of microservices, this could be something to explore (or maybe Uber will add it to it's Open Source stable).

 There is a lot of complex detail here, so strap in when you're ready to read it.

### [Testing Logging in ASPNET Core](https://ardalis.com/testing-logging-in-aspnet-core/)

Many of the posts written by Steve 'Ardalis' Smith are worth revisiting from time to time, and this is no exception. Even though it was last updated in 2019, it provides not only great information on testing .Net logging, but the strategies are solid for testing any dependency that uses either an extension method or even a static class/method. If you're in the .Net realm, or use a language that supports these types of constructs, it's worth a few minutes to refresh.
