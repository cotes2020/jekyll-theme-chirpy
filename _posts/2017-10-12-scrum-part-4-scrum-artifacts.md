---
title: 'Scrum Part 4 - Scrum Artifacts'
date: 2017-10-12T21:43:36+02:00
author: Wolfgang Ofner
categories: [Miscellaneous]
tags: [Agile, Scrum]
---
Scrum has a set of predefined Scrum artifacts. In this post, I will give you an overview of the different roles and talk about my own experience.

## **Product Backlog**

  * Items in the Product Backlog can be called features, backlog item, change requests, bugs, requirements, defects…
  * Contains all features of the product
  * The Product Backlog is owned by the Product Owner
  * Every item has a description, an order and an effort estimation
  * Constantly changing (developers implement features or new features will be added)
  * Items on the top are higher prioritized and therefore described in more detail

The Product Owner and the team should work together to prioritize the tasks in the right order and give each other feedback if a feature needs a better description. An example for making a prioritize change by the team could be a feature which is highly technical and also really important to be implemented. If it’s too technical the PO probably doesn’t understand it or can’t see its importance. In this case, the team has to explain the details to the PO to get him on the same page.

The Product Backlog should be managed with a tool like TFS. If you use TFS as your source control too, it&#8217;s possible to link a feature with a commit to increase the tractability of changes. In the following, you can see the Product Backlog in TFS.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2017/10/Product-Backlog.png"><img loading="lazy" src="/assets/img/posts/2017/10/Product-Backlog.png" alt="Product Backlog" /></a>
  
  <p>
    <a href="https://msdnshared.blob.core.windows.net/media/MSDNBlogsFS/prod.evol.blogs.msdn.com/CommunityServer.Blogs.Components.WeblogFiles/00/00/00/30/15/metablogapi/0250.Figure-1_59B7BEE6.png" target="_blank" rel="noopener">Source</a>
  </p>
</div>

## **Sprint Backlog**

The Sprint Backlog is a subset of selected Backlog Items for the current sprint and also a plan for implementing them into a product. The remaining work should be summed up at least daily (Burndown chart). The Sprint Backlog belongs only to the development team and shows all necessary work to reach the sprint goal. The items in the Backlog can change several times within a sprint since the team gains more experience about the work needed to be done during the sprint. The Sprint Backlog should be seen as a forecast on what will be done instead of a commitment.

Personally, I like to display the Sprint Backlog on a whiteboard. On the following screenshot, you can see my current whiteboard. I whitened business relevant delicate tasks.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2017/10/Sprint-Backlog-Board.jpg"><img loading="lazy" src="/assets/img/posts/2017/10/Sprint-Backlog-Board.jpg" alt="Scrum Artifacts Sprint Backlog Board" /></a>
  
  <p>
    Kanban Board for the Sprint Backlog
  </p>
</div>

### My Kanban Board for the Sprint Backlog

On our board we have the following columns:

  * **Backlog:** current sprint Backlog Items
  * **Prioritized:** important tasks which should be implemented soon
  * **Development In Progress:** features which are currently implemented
  * **Development Complete:** features which are implemented but not forwarded for the review yet
  * **Team To Verify In Progress:** according to our Definition of Done, an implementation has to be reviewed by a second team member
  * **Team To Verify Complete:** the feature was reviewed and can be deployed to the test server
  * **Customer  To Verify In Progress:** the feature is deployed on the test server
  * **Customer To Verify Complete:** the customer reviewed the feature and gave his ok
  * **Done:** the feature is deployed onto the live server

We also have a small area where we can write down impediments or tasks we want to improve in the current sprint.

Every team member has a magnet with their picture on it. So, you can easily see who is working on what. The flash symbol means that the work on this feature is blocked. Next to the white board we hung our Definition of Done. It’s good to have a look at it every now and then.

## **Sprint Goal**

The Sprint Goal is set during the Sprint Planning. It’s a decision made by the development team and Product Owner after selecting and estimating the tasks for the sprint.

## **Definition of Done**

When is a developer done with his task? When the code is checked in? Code committed? Product shipped?

The answer is: it depends. It depends on the product you are working on and on the team. If you are working on a simple calculator then the definition of done won’t be too strict. If you are working on a software which controls parts of a spaceship your definition of done will be long and strict.

I worked on a team where done meant that the programmer said that he is done. Committing the code, writing tests, customer feedback… Nothing needed. Only the word of the developer. I guess it’s no surprise to you that the quality of the shipped product wasn’t too high and there were plenty of bugs.

For me, done means done. A feature or bug is done when the code is checked in into the version control, all tests passed (with a high code coverage), the branch is merged into the develop branch, the CI build is green and the customer or product owner gives his ok.

With all these criteria, a high code quality and customer satisfaction can be achieved.

Every team member must agree to the Definition of Done. This definition can vary from company to company. The Definition of Done is also a standard which ensures a certain level of quality.

A piece of software can only be in one of two states: done or not done. There is no almost done.

## **Burndown Chart**

The Burndown Chart displays the number of remaining features with the remaining time. Tools like TFS also show you an ideal progress which is linear from the start to the end with no work left. Such a linear progress is only in theory possible but the actual progress should be around the theoretical progress. Often the actual progress is like a wave, sometimes over this ideal trend and sometimes below it. On the following screenshot, you can see such a burndown chart from the TFS.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2017/10/Burndown-chart.png"><img loading="lazy" size-full" src="/assets/img/posts/2017/10/Burndown-chart.png" alt="Scrum Artifacts Burndown chart" /></a>
  
  <p>
    <a href="https://docs.microsoft.com/en-us/vsts/work/scrum/_img/alm_sb_introhealthychart.png" target="_blank" rel="noopener">Source</a>
  </p>
</div>

## **Increment**

The increment is the result of the sprint, the software delivered at the end of the sprint

The Product Owner determines what to do with it. Maybe it needs refinement, maybe it can be shipped. The result of the sprint only contains finished work. Features which haven’t met the Definition of Done are not included.

&nbsp;

Next: <a href="/scrum-part-5-meetings/" target="_blank" rel="noopener">Scrum Part 5 &#8211; Meetings</a>

Previous: <a href="/scrum-part-3-scrum-roles/" target="_blank" rel="noopener">Scrum Part 3 &#8211; Scrum roles</a>