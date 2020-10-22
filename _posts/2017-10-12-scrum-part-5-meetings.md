---
title: 'Scrum Part 5 - Meetings'
date: 2017-10-12T22:16:42+02:00
author: Wolfgang Ofner
categories: [Miscellaneous]
tags: [Agile, Scrum]
---
Every Scrum meeting is time-boxed. But what does time-boxed mean? I will explain this term and afterwards, I will present all Scrum meetings.

## **Time-boxing**

Every meeting is a time-boxed event. Time-boxed means that there is a maximum duration. This means if a meeting is scheduled to last two hours, then the meeting is over after latest two hours. Time-boxed doesn’t allow to extend the meeting, but it’s possible to finish the meeting early.

Due to the time boxing, the participants focus better on getting the best results in the given time.

## **Sprint**

The Sprint is a time-boxed event which consists of different events. A sprint usually lasts between 2 &#8211; 4 weeks. An advantage of a short sprint is that the risk of failing is limited. It’s unrealistic to plan 5 or 6 weeks ahead. This long timespan results more in guessing than planning. Planning only 2 &#8211; 4 weeks ahead mitigates the risk of not being able to produce a shippable product at the end of the sprint.

Limited time helps to create and keep the focus on the work currently needed to be done. Seeing a goal which will be reached in 2 weeks is way easier than seeing a goal which will be reached in 6 weeks. A nice example is the weather forecast. The weather can be predicted maybe a couple of days in advance more or less precise. If you look at the forecast for ten days later it’s more a recommendation than a forecast.

A sprint is a learning loop. It consists of a plan, do, check, act or learn, build and measure. No matter what definition you prefer, try, learn, try again. For example, a two-week long sprint means that a new version of the software will be delivered 26 times a year. This means 26 times feedback from the customer and 26 chances to improve the own process within the development team.

It’s important to note that to change bad things you actually have to change. Changes can be hard for a team but they should be seen as a chance to improve the building process which results in a happier customer.

[<img loading="lazy" class="wp-image-186 aligncenter" src="/assets/img/posts/2017/10/Learning-loop.jpg" alt="Scrum sprint learning loop" width="600" height="398"/>](/assets/img/posts/2017/10/Learning-loop.jpg)

## Velocity

The productivity can be measured in velocity. The velocity is measured every sprint. After some sprints, the team will be able to make a forecast on how much work they can get done in the next sprint. There is no unit for the velocity. This could be finished features, story points or whatever fits your team best.

If the team finished on average between 50 and 60 units over the last five sprint then it is likely that the team will finish between 50 and 60 units in the next sprint too. This helps you to choose the right amount of features for the sprint. If you estimate your features will take you 25 units to finish then the team either estimated too positively or took too little features. If this occurs the planning should be revised. It’s also unlikely that the team can suddenly finish 80 units in one sprint.

It’s common to start and finish each sprint on the same day. I like to start on a Monday and finish on a Friday. So the team comes back hopefully motivated and fresh from the weekend and goes into the weekend with a finished sprint.

## **Sprint Planning**

The Sprint Planning is a time-boxed meeting which should last max. 2 hours for every week of the sprint. The whole team comes together to talk about uncertainties and plan the tasks for the coming sprint. It is also possible to invite external people, for example, domain experts who can help by clarifying some problems. During the Sprint Planning, the Sprint Backlog and Sprint goals are created. To plan a sprint, an ordered Product Backlog is needed. Otherwise, the team can’t decide which tasks are important and which are not. Therefore the PO should also attend the Sprint Planning.

The team takes the tasks one after another and discusses them. If they think that they can implement the task in the sprint, then they will add it to the Sprint Backlog. After the Sprint Backlog is finished, the team will talk about the plan on how to deliver the chosen tasks. They also set a Sprint goal. At the end of this discussion, the team will be able to explain to the PO what the goal is for this Sprint and how they want to achieve it.

## **Planning Poker**

The Planning Poker is a technique to estimate the effort of a feature. First, the Product Owner explains a feature. Then the development teams discuss the feature and if necessary, asks questions. After this discussion, the actual planning takes place. Everyone selects a card (a physical card or in an app) and then all team member present their selection at the same time. Let me give you an example why it’s important to reveal the card at the same time. For example, the lead developer presents his estimation first. His estimation will influence every team member and their results will be close to his. If every estimation is presented at the same time, the range of estimates will be wider.

### Unit of work

As a unit of work, you can choose whatever fits your team best. I like using story points with the values of Fibonacci (1, 2, 3, 5, 8, 13, 21 and 40). 40 is not a part of Fibonacci but if a feature takes 40 story points, it’s too big and needs breaking into smaller pieces. The reason why I like these values is that I can see that 21 one is a bit less than double of a 13. It is way harder to differentiate between 13 and 15. If everyone estimated the same amount for the feature the round is finished and the result will be put onto the Sprint Backlog Item.

If the outcome is different, the team has to discuss why the member chose different values. For example, if the outcome is 3, 8, 8, 13, 21 then the person who selected 3 and 21 should explain why they chose this number. This does not mean that they are wrong at all. This could mean that they haven’t understood everything or maybe the whole team didn’t see a specific detail which made the feature look way harder or easier. After the discussion, the poker round is repeated.

### Choosing an appropriate amount of work

This estimation process is repeated until all items are estimated. After the estimation is finished, every item has a value of story points. This can be mapped to hours of work. But that is not necessary. If your team has a velocity of 50 story points and your Sprint Backlog has items with 49 story points in total, you know that these items will take you the whole sprint to implement. I have to admit that it might take a bit to get used to. Story points are not comparable between two teams. If team A estimates 50 story points and team B estimates 150 story points, it could result in two weeks of work for both teams.

I have also seen teams using T-shirt sizes instead of story points. The sizes are between small and XL. In the end, it will have the same outcome as story points. If this approach works better for your team, use it.

There are plenty of apps for free on the market. I use the one from bbv which can be found in the <a href="https://itunes.apple.com/ch/app/bbv-planning-poker/id1204820215?l=en&mt=8" target="_blank" rel="noopener">App store</a> and in <a href="https://play.google.com/store/apps/details?id=ch.bbv.planningpoker&hl=en" target="_blank" rel="noopener">Google Play</a>.

## **Daily Scrum (Daily Stand-up Meeting)**

The Daily Stand-up Meeting is a daily meeting for and by the development team. The time box is 15 minutes. This meeting serves as a quick check on what’s going on in the team. In this meeting, the team makes a plan for the current day. If the team has some problems which it can’t solve by itself, it’s the Scrum Masters responsibility to help the team to get these problems solved.

Every developer has to answer 3 questions. These questions are:

  * What did I do yesterday?
  * What will I do today?
  * Are there any impediments in my way?

Only one person talks at a time. If there is anything else that needs to be discussed and doesn’t affect the whole team, it should be discussed after the meeting with only the needed people.

## **Sprint Review**

This meeting is conducted at the end of the sprint. The goal is to share the result within the organization or with the stakeholders. Interested people of the organization could be the management but also marketing. The time box is set to 1 hour per week of the sprint. A shorter time span is more likely to attract more people. During the Sprint Review, the team shows the implemented features and also asks about feedback of the attendances. The goal is to get feedback which can result in new Product Backlog Items. These new items will lead to an improved quality and customer happiness.

## **Sprint Retrospective**

The Sprint Retrospective marks the end of the sprint. After the Sprint Retrospective is over, the sprint is over. Attending is the entire Scrum Team (PO, SM and the team). Outsiders are not present in this meeting. The Sprint Retrospective is a chance to reflect over the last sprint and talk about what the team can improve to achieve better results with the next sprint. If the team has issues finding problems or can’t decide which problems need improvement the most, then the Scrum Master should guide the team to a solution.

The meeting is also a chance to talk about the Definition of Done. Maybe the definition needs refinement. This can be discussed within the whole team.

### How my team does the Retrospective

In my team we have a white board with 3 columns:

  * the good
  * the bad
  * information

Every team member prepares post-its and while sticking the post-it to the white board, he says a couple of words about the post it. After all post-its are on the board, we try to group them into three groups. The groups could be improvements, emotions, processes and so on. After we have defined the groups, everyone places two points to the most important topic and one point to the second most important topic. After all points are placed we talk about the category with the most points and try to find ways to improve in the next sprint. It is also possible that the team focuses on good things and decides to try to keep something which went really well at the current high level.

If you hear for the first time about these meetings, you might think that the whole team will spend too much time in meetings and won’t get work done. In my experience when applying Scrum, the team spends less time in meetings. Most meetings are status updates or other coordination tasks which will go away when using Scrum.

&nbsp;

Next: <a href="/scrum-part-6-rules/" target="_blank" rel="noopener">Scrum Part 6 &#8211; Rules</a>

Previous: <a href="/scrum-part-4-scrum-artifacts/" target="_blank" rel="noopener">Scrum Part 4 &#8211; Scrum Artifacts</a>