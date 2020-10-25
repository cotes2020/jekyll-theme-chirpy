---
title: Run the CI Pipeline during a Pull Request
date: 2020-10-17T16:34:44+02:00
author: Wolfgang Ofner
categories: [DevOps]
tags: [Azure Devops, CI, continuous integration, DevOps Pull request policy]
---
In the modern DevOps culture, the goal is to get features as fast as possible into production. Additionally, we have to guarantee that these new features don&#8217;t break anything. To do that, I will show in this post how to protect the master branch with a policy that enforces a pull request (PR) and reviews. To further enhance the quality, I will show <a href="/run-the-ci-pipeline-during-pull-request" target="_blank" rel="noopener noreferrer">how to run the CI pipeline from my last post</a>, which builds the solutions and runs all unit tests.

## Protect the Master Branch with a Pull Request Policy

To create a new policy go to Project settings  &#8211;> Repositories &#8211;> Policies &#8211;> Branch policies and there click the + button.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/08/Create-a-new-Branch-Policy.jpg"><img loading="lazy" src="/assets/img/posts/2020/08/Create-a-new-Branch-Policy.jpg" alt="Create a new Branch Policy" /></a>
  
  <p>
    Create a new Branch Policy
  </p>
</div>

This opens a fly-out where you can select either of the two options. I select &#8220;Protect current and future branches matching a specified pattern&#8221; and enter master as the branch name. This means that this policy is only valid for the master branch. Then click Create.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/08/Add-a-Pull-Request-Policy-for-the-master-branch.jpg"><img loading="lazy" src="/assets/img/posts/2020/08/Add-a-Pull-Request-Policy-for-the-master-branch.jpg" alt="Add a Pull Request Policy for the master branch" /></a>
  
  <p>
    Add a Pull Request Policy for the master branch
  </p>
</div>

This opens the Branch Policies menu where you can configure your pull request.

### Configure the Branch Policy

First, I require one reviewer, allow the requestor to approve their changes, and reset the vote every time new changes are committed. Usually, I don&#8217;t allow the requestor to approve their changes but since I am alone in this demo project I allow it. Microsoft recommends that two reviewers should check the pull request for the highest quality.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/08/Configure-the-minimum-number-of-reviewers.jpg"><img loading="lazy" src="/assets/img/posts/2020/08/Configure-the-minimum-number-of-reviewers.jpg" alt="Configure the minimum number of reviewers" /></a>
  
  <p>
    Configure the minimum number of reviewers
  </p>
</div>

Next, I require every pull request to be linked with a work item. There should never be code changes without a PBI or Bug ticket describing the desired changes. Therefore, this is required.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/08/Check-for-linked-work-items.jpg"><img loading="lazy" src="/assets/img/posts/2020/08/Check-for-linked-work-items.jpg" alt="Check for linked work items" /></a>
  
  <p>
    Check for linked work items
  </p>
</div>

Reviewers provide their feedback with comments, therefore, I require all comments to be resolved before a pull request can be completed. In my projects, always the creator of the comment resolves the comment and not the creator of the PR.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/08/Configure-comment-resolution.jpg"><img loading="lazy" src="/assets/img/posts/2020/08/Configure-comment-resolution.jpg" alt="Configure comment resolution" /></a>
  
  <p>
    Configure comment resolution
  </p>
</div>

Companies have different merge strategies. Some use squash merges, some do rebase and some do just basic merges. It is good to have a merge strategy and limit the pull request to your strategy. In my projects, I do squash merges because I don&#8217;t care about all the commits during the development. I only care about the finished feature commit, therefore, I only allow squash merges.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/08/Limit-the-merge-types.jpg"><img loading="lazy" src="/assets/img/posts/2020/08/Limit-the-merge-types.jpg" alt="Limit the merge types" /></a>
  
  <p>
    Limit the merge types
  </p>
</div>

### Configure automatic Builds

Now we come to the most interesting part of the policy. I add a build policy and select the previously created CustomerApi CI pipeline.  You can find the post <a href="/build-net-core-in-ci-pipeline-in-azure-devops" target="_blank" rel="noopener noreferrer">here</a>. I set /CustomerApi/* as path filter. The automatic trigger starts the build every time changes are committed inside the CustomerApi folder and the build expires after 12 hours. This means if the pull request is not completed within 12 hours, the build has to be triggered again.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/08/Add-a-build-policy-for-the-CustomerApi-to-the-Pull-Request.jpg"><img loading="lazy" src="/assets/img/posts/2020/08/Add-a-build-policy-for-the-CustomerApi-to-the-Pull-Request.jpg" alt="Add a build policy for the CustomerApi to the Pull Request" /></a>
  
  <p>
    Add a build policy for the CustomerApi to the Pull Request
  </p>
</div>

Add another build policy for the OrderApi and enter /OrderApi/* as path filter. Click on Save and the policy is configured and created.

## Make changes to the Code

I added a new unit test, commit the changes to the master branch, and push the changes. Due to the policy on the master branch, I am not allowed to push changes directly to the master branch, as seen on the following screenshot.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/08/Pushing-directly-to-master-is-not-allowed.jpg"><img loading="lazy" src="/assets/img/posts/2020/08/Pushing-directly-to-master-is-not-allowed.jpg" alt="Pushing directly to master is not allowed" /></a>
  
  <p>
    Pushing directly to master is not allowed
  </p>
</div>

Since the master branch is protected, I have to create a feature branch. I name this branch addunittest and push the changes to Azure DevOps.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/08/Pushing-a-new-branch.jpg"><img loading="lazy" src="/assets/img/posts/2020/08/Pushing-a-new-branch.jpg" alt="Pushing a new branch" /></a>
  
  <p>
    Pushing a new branch
  </p>
</div>

In Azure DevOps under Repos &#8211;> Files, you can see that Azure DevOps registered the changes and already suggest to create a new PR. Click on Create a pull request and you will get into a new window.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/08/Create-a-new-Pull-Request.jpg"><img loading="lazy" src="/assets/img/posts/2020/08/Create-a-new-Pull-Request.jpg" alt="Create a new Pull Request" /></a>
  
  <p>
    Create a new Pull Request
  </p>
</div>

## Create a new Pull Request

Add a title, and optionally a description, reviewers, and work items. I like to have 1-3 sentences in the description to explain what you did. As the title, I usually use the PBI number and the PBI title (not in this example though).

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/08/New-Pull-Request.jpg"><img loading="lazy" src="/assets/img/posts/2020/08/New-Pull-Request.jpg" alt="New Pull Request" /></a>
  
  <p>
    New Pull Request
  </p>
</div>

After the pull request is created, the build will kick off immediately, and also all other required policies will be checked. As you can see on the following screenshot, the build failed due to a failing test, no work item is linked and not all comments are resolved. Therefore, I can&#8217;t complete the PR.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/08/Required-checks-failed.jpg"><img loading="lazy" src="/assets/img/posts/2020/08/Required-checks-failed.jpg" alt="Required checks failed" /></a>
  
  <p>
    Required checks failed
  </p>
</div>

I fixed my unit test, added a link to the PB,I and fixed the suggested changes from the comment. The comment creator resolved the comment and this enabled me to complete the pull request. On the following screenshot, you can see that I also changed the title of the PR to have to PBI number and title in it.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/08/All-checks-passed.jpg"><img loading="lazy" src="/assets/img/posts/2020/08/All-checks-passed.jpg" alt="All checks passed" /></a>
  
  <p>
    All checks passed
  </p>
</div>

### Complete the Pull Request

When you click on Complete, you can select a merge type. Since I restricted the merge strategy to squash commit only, I can&#8217;t select any other strategy.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/08/Only-Squash-commit-is-allowed-by-the-Pull-Request-policy.jpg"><img loading="lazy" src="/assets/img/posts/2020/08/Only-Squash-commit-is-allowed-by-the-Pull-Request-policy.jpg" alt="Only Squash commit is allowed by the Pull Request policy" /></a>
  
  <p>
    Only Squash commit is allowed by the Pull Request policy
  </p>
</div>

The definition of done in my projects is that the PBI is set to done when the pull request is finished (because we deploy the feature automatically to prod when thePR is completed). Additionally, I select to delete my branch after merging.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/08/Complete-the-Pull-Request.jpg"><img loading="lazy" src="/assets/img/posts/2020/08/Complete-the-Pull-Request.jpg" alt="Complete the Pull Request" /></a>
  
  <p>
    Complete the Pull Request
  </p>
</div>

The PR creator can also select auto-complete to complete the pull request automatically when all required checks are OK. After the merge to master is completed, the CI pipeline automatically kicks off a build of the master branch.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/08/The-master-branch-trigger-a-CI-build.jpg"><img loading="lazy" src="/assets/img/posts/2020/08/The-master-branch-trigger-a-CI-build.jpg" alt="The master branch trigger a CI build" /></a>
  
  <p>
    The master branch trigger a CI build
  </p>
</div>

## Conclusion

In this post, I explained how to protect the master branch from changes in Azure DevOps. I showed how to add a branch policy to the master branch in Azure DevOps and also how to run a build process to check if the solution compiles and if all tests run successfully.

You can find the code of this demo on <a href="https://github.com/WolfgangOfner/MicroserviceDemo" target="_blank" rel="noopener noreferrer">GitHub</a>.