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

<div id="attachment_2327" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/08/Create-a-new-Branch-Policy.jpg"><img aria-describedby="caption-attachment-2327" loading="lazy" class="wp-image-2327" src="/wp-content/uploads/2020/08/Create-a-new-Branch-Policy.jpg" alt="Create a new Branch Policy" width="700" height="352" /></a>
  
  <p id="caption-attachment-2327" class="wp-caption-text">
    Create a new Branch Policy
  </p>
</div>

This opens a fly-out where you can select either of the two options. I select &#8220;Protect current and future branches matching a specified pattern&#8221; and enter master as the branch name. This means that this policy is only valid for the master branch. Then click Create.

<div id="attachment_2326" style="width: 478px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/08/Add-a-Pull-Request-Policy-for-the-master-branch.jpg"><img aria-describedby="caption-attachment-2326" loading="lazy" class="size-full wp-image-2326" src="/wp-content/uploads/2020/08/Add-a-Pull-Request-Policy-for-the-master-branch.jpg" alt="Add a Pull Request Policy for the master branch" width="468" height="305" /></a>
  
  <p id="caption-attachment-2326" class="wp-caption-text">
    Add a Pull Request Policy for the master branch
  </p>
</div>

This opens the Branch Policies menu where you can configure your pull request.

### Configure the Branch Policy

First, I require one reviewer, allow the requestor to approve their changes, and reset the vote every time new changes are committed. Usually, I don&#8217;t allow the requestor to approve their changes but since I am alone in this demo project I allow it. Microsoft recommends that two reviewers should check the pull request for the highest quality.

<div id="attachment_2330" style="width: 614px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/08/Configure-the-minimum-number-of-reviewers.jpg"><img aria-describedby="caption-attachment-2330" loading="lazy" class="size-full wp-image-2330" src="/wp-content/uploads/2020/08/Configure-the-minimum-number-of-reviewers.jpg" alt="Configure the minimum number of reviewers" width="604" height="298" /></a>
  
  <p id="caption-attachment-2330" class="wp-caption-text">
    Configure the minimum number of reviewers
  </p>
</div>

Next, I require every pull request to be linked with a work item. There should never be code changes without a PBI or Bug ticket describing the desired changes. Therefore, this is required.

<div id="attachment_2329" style="width: 516px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/08/Check-for-linked-work-items.jpg"><img aria-describedby="caption-attachment-2329" loading="lazy" class="size-full wp-image-2329" src="/wp-content/uploads/2020/08/Check-for-linked-work-items.jpg" alt="Check for linked work items" width="506" height="219" /></a>
  
  <p id="caption-attachment-2329" class="wp-caption-text">
    Check for linked work items
  </p>
</div>

Reviewers provide their feedback with comments, therefore, I require all comments to be resolved before a pull request can be completed. In my projects, always the creator of the comment resolves the comment and not the creator of the PR.

<div id="attachment_2331" style="width: 530px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/08/Configure-comment-resolution.jpg"><img aria-describedby="caption-attachment-2331" loading="lazy" class="size-full wp-image-2331" src="/wp-content/uploads/2020/08/Configure-comment-resolution.jpg" alt="Configure comment resolution" width="520" height="213" /></a>
  
  <p id="caption-attachment-2331" class="wp-caption-text">
    Configure comment resolution
  </p>
</div>

Companies have different merge strategies. Some use squash merges, some do rebase and some do just basic merges. It is good to have a merge strategy and limit the pull request to your strategy. In my projects, I do squash merges because I don&#8217;t care about all the commits during the development. I only care about the finished feature commit, therefore, I only allow squash merges.

<div id="attachment_2332" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/08/Limit-the-merge-types.jpg"><img aria-describedby="caption-attachment-2332" loading="lazy" class="wp-image-2332" src="/wp-content/uploads/2020/08/Limit-the-merge-types.jpg" alt="Limit the merge types" width="700" height="210" /></a>
  
  <p id="caption-attachment-2332" class="wp-caption-text">
    Limit the merge types
  </p>
</div>

### Configure automatic Builds

Now we come to the most interesting part of the policy. I add a build policy and select the previously created CustomerApi CI pipeline.  You can find the post <a href="/build-net-core-in-ci-pipeline-in-azure-devops" target="_blank" rel="noopener noreferrer">here</a>. I set /CustomerApi/* as path filter. The automatic trigger starts the build every time changes are committed inside the CustomerApi folder and the build expires after 12 hours. This means if the pull request is not completed within 12 hours, the build has to be triggered again.

<div id="attachment_2506" style="width: 372px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/08/Add-a-build-policy-for-the-CustomerApi-to-the-Pull-Request.jpg"><img aria-describedby="caption-attachment-2506" loading="lazy" class="wp-image-2506" src="/wp-content/uploads/2020/08/Add-a-build-policy-for-the-CustomerApi-to-the-Pull-Request.jpg" alt="Add a build policy for the CustomerApi to the Pull Request" width="362" height="700" /></a>
  
  <p id="caption-attachment-2506" class="wp-caption-text">
    Add a build policy for the CustomerApi to the Pull Request
  </p>
</div>

Add another build policy for the OrderApi and enter /OrderApi/* as path filter. Click on Save and the policy is configured and created.

## Make changes to the Code

I added a new unit test, commit the changes to the master branch, and push the changes. Due to the policy on the master branch, I am not allowed to push changes directly to the master branch, as seen on the following screenshot.

<div id="attachment_2334" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/08/Pushing-directly-to-master-is-not-allowed.jpg"><img aria-describedby="caption-attachment-2334" loading="lazy" class="wp-image-2334" src="/wp-content/uploads/2020/08/Pushing-directly-to-master-is-not-allowed.jpg" alt="Pushing directly to master is not allowed" width="700" height="131" /></a>
  
  <p id="caption-attachment-2334" class="wp-caption-text">
    Pushing directly to master is not allowed
  </p>
</div>

Since the master branch is protected, I have to create a feature branch. I name this branch addunittest and push the changes to Azure DevOps.

<div id="attachment_2335" style="width: 636px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/08/Pushing-a-new-branch.jpg"><img aria-describedby="caption-attachment-2335" loading="lazy" class="size-full wp-image-2335" src="/wp-content/uploads/2020/08/Pushing-a-new-branch.jpg" alt="Pushing a new branch" width="626" height="85" /></a>
  
  <p id="caption-attachment-2335" class="wp-caption-text">
    Pushing a new branch
  </p>
</div>

In Azure DevOps under Repos &#8211;> Files, you can see that Azure DevOps registered the changes and already suggest to create a new PR. Click on Create a pull request and you will get into a new window.

<div id="attachment_2336" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/08/Create-a-new-Pull-Request.jpg"><img aria-describedby="caption-attachment-2336" loading="lazy" class="wp-image-2336" src="/wp-content/uploads/2020/08/Create-a-new-Pull-Request.jpg" alt="Create a new Pull Request" width="700" height="100" /></a>
  
  <p id="caption-attachment-2336" class="wp-caption-text">
    Create a new Pull Request
  </p>
</div>

## Create a new Pull Request

Add a title, and optionally a description, reviewers, and work items. I like to have 1-3 sentences in the description to explain what you did. As the title, I usually use the PBI number and the PBI title (not in this example though).

<div id="attachment_2337" style="width: 690px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/08/New-Pull-Request.jpg"><img aria-describedby="caption-attachment-2337" loading="lazy" class="wp-image-2337" src="/wp-content/uploads/2020/08/New-Pull-Request.jpg" alt="New Pull Request" width="680" height="700" /></a>
  
  <p id="caption-attachment-2337" class="wp-caption-text">
    New Pull Request
  </p>
</div>

After the pull request is created, the build will kick off immediately, and also all other required policies will be checked. As you can see on the following screenshot, the build failed due to a failing test, no work item is linked and not all comments are resolved. Therefore, I can&#8217;t complete the PR.

<div id="attachment_2339" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/08/Required-checks-failed.jpg"><img aria-describedby="caption-attachment-2339" loading="lazy" class="wp-image-2339" src="/wp-content/uploads/2020/08/Required-checks-failed.jpg" alt="Required checks failed" width="700" height="323" /></a>
  
  <p id="caption-attachment-2339" class="wp-caption-text">
    Required checks failed
  </p>
</div>

I fixed my unit test, added a link to the PB,I and fixed the suggested changes from the comment. The comment creator resolved the comment and this enabled me to complete the pull request. On the following screenshot, you can see that I also changed the title of the PR to have to PBI number and title in it.

<div id="attachment_2341" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/08/All-checks-passed.jpg"><img aria-describedby="caption-attachment-2341" loading="lazy" class="wp-image-2341" src="/wp-content/uploads/2020/08/All-checks-passed.jpg" alt="All checks passed" width="700" height="255" /></a>
  
  <p id="caption-attachment-2341" class="wp-caption-text">
    All checks passed
  </p>
</div>

### Complete the Pull Request

When you click on Complete, you can select a merge type. Since I restricted the merge strategy to squash commit only, I can&#8217;t select any other strategy.

<div id="attachment_2344" style="width: 474px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/08/Only-Squash-commit-is-allowed-by-the-Pull-Request-policy.jpg"><img aria-describedby="caption-attachment-2344" loading="lazy" class="size-full wp-image-2344" src="/wp-content/uploads/2020/08/Only-Squash-commit-is-allowed-by-the-Pull-Request-policy.jpg" alt="Only Squash commit is allowed by the Pull Request policy" width="464" height="405" /></a>
  
  <p id="caption-attachment-2344" class="wp-caption-text">
    Only Squash commit is allowed by the Pull Request policy
  </p>
</div>

The definition of done in my projects is that the PBI is set to done when the pull request is finished (because we deploy the feature automatically to prod when thePR is completed). Additionally, I select to delete my branch after merging.

<div id="attachment_2345" style="width: 481px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/08/Complete-the-Pull-Request.jpg"><img aria-describedby="caption-attachment-2345" loading="lazy" class="size-full wp-image-2345" src="/wp-content/uploads/2020/08/Complete-the-Pull-Request.jpg" alt="Complete the Pull Request" width="471" height="342" /></a>
  
  <p id="caption-attachment-2345" class="wp-caption-text">
    Complete the Pull Request
  </p>
</div>

The PR creator can also select auto-complete to complete the pull request automatically when all required checks are OK. After the merge to master is completed, the CI pipeline automatically kicks off a build of the master branch.

<div id="attachment_2346" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/08/The-master-branch-trigger-a-CI-build.jpg"><img aria-describedby="caption-attachment-2346" loading="lazy" class="wp-image-2346" src="/wp-content/uploads/2020/08/The-master-branch-trigger-a-CI-build.jpg" alt="The master branch trigger a CI build" width="700" height="401" /></a>
  
  <p id="caption-attachment-2346" class="wp-caption-text">
    The master branch trigger a CI build
  </p>
</div>

## Conclusion

In this post, I explained how to protect the master branch from changes in Azure DevOps. I showed how to add a branch policy to the master branch in Azure DevOps and also how to run a build process to check if the solution compiles and if all tests run successfully.

You can find the code of this demo on <a href="https://github.com/WolfgangOfner/MicroserviceDemo" target="_blank" rel="noopener noreferrer">GitHub</a>.