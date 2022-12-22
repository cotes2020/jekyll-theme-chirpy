---
title: Day at Dynamo Hack
description: Experience on a hack day. An entry of experience on North East Dynamo hack day
tags: ["d17hack", "dynamo", "hackathon", "northeast"]
category: ["experience"]
date: 2017-07-17
permalink: 'random/day-at-dynamo-hack/'
---


# Introduction
On 3rd July, I got an excellent opportunity to attend Dynamo Hackathon. It all started during the second week of June when I received an email from my organisation about the Dynamo summit. I registered my interest and, soon I received a few more emails and an invite to join Dynamo slack. With numerous conversations over slack, emails and Skype, I came to know that I would be placed in a small team of 5-6 people and would be expected to produce a minimal viable product(MVP) within 8 hours. The MVP would then be judged by a panel from Dynamo. The time limit of 8 hours sounds too aggressive. Even though this was not my first hackathon, I was growing anxious. The theme of the hackathon was to promote and showcase the technical skills and technical capabilities within North East(NE).

![Dynamo Banner](https://raw.githubusercontent.com/Gaur4vGaur/traveller/master/images/random/2017-07-17-day-at-dynamo-hack/dynamo.png)

Initially, I thought that I would be working with the colleagues of my organisation, but soon I realised that the hackathon composed team picked people from different organisations, and with multiple skills. Being an introvert slightly made me more anxious, but at the same time, I was excited to be a part of a talented team.

The organizers provided me with slack and emails for other members of the team and we started introducing each other. The team started to conceptualise the idea of what we are going to build on the day. After multiple rounds of discussion, we settled to create a visualisation of the skill gap in NE.

### The Day
Although BBC had predicted rain on the day, it turns out to be a lovely sunny day. It was a scenic drive from Newcastle to Durham. Thanks to one of my team member, who offered me a lift from Newcastle Central Station. We reach the historic city of Durham by 7:45 AM and joined other team members at Durham University.

As planned earlier, we started to develop a visualisation portal to aid new graduates or job seekers to hunt jobs within a specific region. The visualisation would show various companies available within a specific region and their job requirements. Along with the job vacancies, the portal would link to free/paid online training available to bridge the skill gap.

![Team](https://raw.githubusercontent.com/Gaur4vGaur/traveller/master/images/random/2017-07-17-day-at-dynamo-hack//team2.png)

### The Hack
After rigorous brainstorming, we chose to subscribe to Indeed API to fetch the job requirements and Google API/MS tableau to visualise the same. The team decided to fetch the data from API and populate it in a local instance of Mongo. The plan was to utilise Mongo DB to create some more stats and trends if we left with some extra time. We broke the application into 3 components:
* Component 1 - batch script that would subscribe Indeed API and populate local Mongo Instance,
* Component 2 - a front end that would visualize the data as the per the user requests, and,
* Component 3 - a Scala backend that would query local Mongo instance to provide filtered results to the front end

The great thing about the team was that we all had different strengths, and we worked incredibly well together. After a few initial technical hiccups, all three components were available. It was funny to observe how quickly time passes when you are surrounded by determined and equally caffeinated people.

![Team](https://raw.githubusercontent.com/Gaur4vGaur/traveller/master/images/random/2017-07-17-day-at-dynamo-hack//team.png)

### The Finale
We were all set to integrate the components, but we hit our biggest roadblock. Among all of us, two of us were Ubuntu, one was Linux Mint, another one was Windows 10 and the last one was macOS. We were a couple of hours away from the demo and there was no single machine that could run all three components. By random selection and trust in Steve Jobs, we all chose to work/fix on macOS. We managed to complete our MVP when the clock turned red to show the last 10 min remaining for the hack. The judges came by and appreciated the effort

It was certainly an event full of learning, and I enjoyed it a lot. All the teams came up with some amazing MVPs. I will look forward to Dynamo arranging such events in the future.

