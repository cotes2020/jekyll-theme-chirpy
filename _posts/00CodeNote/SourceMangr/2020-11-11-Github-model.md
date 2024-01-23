---
title: GitHub - A successful Git branching model
date: 2020-11-11 11:11:11 -0400
categories: [00CodeNote, SourceMagr]
tags: [git]
toc: true
image:
---

- [A successful Git branching model](#a-successful-git-branching-model)
  - [The main branches](#the-main-branches)
  - [Supporting branches](#supporting-branches)
    - [Feature branches](#feature-branches)
    - [Release branches](#release-branches)
    - [Hotfix branches](#hotfix-branches)

---

## A successful Git branching model

[link](http://nvie.com/posts/a-successful-git-branching-model/)

![model](https://i.imgur.com/4O04y4k.png)

<img alt="pic" src="https://i.imgur.com/Eq8uSel.png" width="500">

---


### The main branches

<img alt="pic" src="https://i.imgur.com/zSM149w.png" width="300">

- master
  - consider origin/master to be the main branch
  - the source code of HEAD always reflects a <font colore=red> production-ready state </font>
- develop
  - consider origin/develop to be the main branch
  - where the source code of HEAD always reflects a state with the <font colore=red> latest delivered development changes for the next release </font>
    - Some would call this the “integration branch”.
    - This is where any automatic nightly builds are built from.
  - When the source code in the develop branch reaches a stable point and is ready to be released
    - all of the changes should be merged back into master somehow
    - and then tagged with a release number.


---


### Supporting branches

uses a variety of supporting branches
- to aid parallel development between team members,
- ease tracking of features,
- prepare for production releases and to assist in quickly fixing live production problems.

Unlike the main branches, these branches always have a limited life time, since they will be removed eventually.

The different types of branches may use are:
- Feature branches
- Release branches
- Hotfix branches


Each of these branches have a specific purpose and are bound to strict rules as to which branches may be their originating branch and which branches must be their merge targets. We will walk through them in a minute.


 The branch types are categorized by how we use them.

---

#### Feature branches

<img alt="pic" src="https://i.imgur.com/rZF6qli.png" width="100">

- May branch off from:
  - develop
- Must merge back into:
  - develop
- Branch naming convention:
  - anything except master, develop, release-*, or hotfix-*

Feature/topic branches
- used to develop new features for the upcoming or a distant future release.
- When starting development of a feature, the target release in which this feature will be incorporated may well be unknown at that point.
- The essence of a feature branch is that <font color=blue> it exists as long as the feature is in development </font>
  - but will eventually be
    - merged back into develop (add the new feature to the upcoming release)
    - or discarded (in case of a disappointing experiment).

- Feature branches typically exist in developer repos only, not in origin.

```bash
# -------------- Creating a feature branch
# branch off from the develop branch.
# Switched to a new branch "myfeature"
$ git checkout -b myfeature develop


# -------------- Incorporating a finished feature on develop
# Finished features may be merged into the develop branch to definitely add them to the upcoming release:

# Switched to branch 'develop'
$ git checkout develop

$ git merge --no-ff myfeature
# Updating ea1b82a..05e9557
# (Summary of changes)

# Deleted branch myfeature (was 05e9557).
$ git branch -d myfeature

$ git push origin develop
```

The --no-ff flag
- causes the merge to always create a new commit object, even if the merge could be performed with a fast-forward.
- This avoids losing information about the historical existence of a feature branch and groups together all commits that together added the feature.
- Compare:


<img alt="pic" src="https://i.imgur.com/XhtCU90.png" width="400">

---


#### Release branches

<img alt="pic" src="https://i.imgur.com/GnPThYo.png" width="400">

- May branch off from:
  - develop
- Must merge back into:
  - develop and master
- Branch naming convention:
  - release-*


Release branches support preparation of a new production release.
- They allow for last-minute dotting of i’s and crossing t’s.
- Furthermore, they allow for minor bug fixes and preparing meta-data for a release (version number, build dates, etc.).
- By doing all of this work on a release branch, the develop branch is cleared to receive features for the next big release.

The key moment to branch off a new release branch from develop is when develop reflects the desired state of the new release.
- all features targeted for the release must be merged in to develop at this point
- All features targeted at future releases may not
  - they must wait until after the release branch is branched off.
- It is exactly at the start of a release branch that the upcoming release gets assigned a version number—not any earlier.
- Up until that moment, the develop branch reflected changes for the “next release”, but it is unclear whether that “next release” will eventually become 0.3 or 1.0, until the release branch is started.
- That decision is made on the start of the release branch and is carried out by the project’s rules on version number bumping.


This new branch may exist there for a while
- until the release may be rolled out definitely.
- During that time, bug fixes may be applied in this branch (rather than on the develop branch).
- Adding large new features here is strictly prohibited. They must be merged into develop, and therefore, wait for the next big release.


```bash
# -------------- Creating a release branch
# Release branches are created from the develop branch.
# For example
# version 1.1.5 is the current production release and we have a big release coming up.
# The state of develop is ready for the “next release” version 1.2 (rather than 1.1.6 or 2.0).
# So we branch off and give the release branch a name reflecting the new version number:

# Switched to a new branch "release-1.2"
$ git checkout -b release-1.2 develop

# Files modified successfully, version number bumped to 1.2.
# bump-version.sh: a fictional shell script, changes some files to reflect the new version, the bumped version number is committed.
$ ./bump-version.sh 1.2

$ git commit -a -m "Bumped version number to 1.2"
# [release-1.2 74d9424] Bumped version number to 1.2
# 1 files changed, 1 insertions(+), 1 deletions(-)




# -------------- Finishing a release branch
# When the state of the release branch is ready to become a real release, some actions need to be carried out.
# 1. the release branch is merged into master
#    (since every commit on master is a new release by definition, remember).
# 2. commit on master must be tagged for easy future reference to this historical version.
# 3. the changes made on the release branch need to be merged back into develop
#    so the future releases also contain these bug fixes.

# -------------- the release branch is merged into master
# Switched to branch 'master'
$ git checkout master

$ git merge --no-ff release-1.2
# Merge made by recursive.
# (Summary of changes)

# The release is now done, and tagged for future reference.
$ git tag -a 1.2

# might as well want to use the -s or -u <key> flags to sign your tag cryptographically.


# -------------- keep the changes made in the release branch, merge back into develop
# Switched to branch 'develop'
$ git checkout develop

$ git merge --no-ff release-1.2
# Merge made by recursive.
# (Summary of changes)



# -------------- the release branch may be removed, since we don’t need it anymore:
# Deleted branch release-1.2 (was ff452fe).
$ git branch -d release-1.2
```

---


#### Hotfix branches


<img alt="pic" src="https://i.imgur.com/2OvkJZO.png" width="400">

- May branch off from:
  - master
- Must merge back into:
  - develop and master
- Branch naming convention:
  - hotfix-*

Hotfix branches
- very much like release branches in that they are also meant to prepare for a new production release, albeit unplanned.
- They arise from the necessity to act immediately upon an undesired state of a live production version.
- When a critical bug in a production version must be resolved immediately, a hotfix branch may be branched off from the corresponding tag on the master branch that marks the production version.
- <font color=red> The essence </font> is that
  - work of team members (on the develop branch) can continue
  - while another person is preparing a quick production fix.

```bash

# -------------- Creating the hotfix branch
# Hotfix branches are created from the master branch.
# For example
# version 1.2 is the current production release running live
# causing troubles due to a severe bug.
# But changes on develop are yet unstable.
# We may then branch off a hotfix branch and start fixing the problem:

# Switched to a new branch "hotfix-1.2.1"
$ git checkout -b hotfix-1.2.1 master

# Files modified successfully, version bumped to 1.2.1.
# Don’t forget to bump the version number after branching off!
$ ./bump-version.sh 1.2.1

$ git commit -a -m "Bumped version number to 1.2.1"
# [hotfix-1.2.1 41e61bb] Bumped version number to 1.2.1
# 1 files changed, 1 insertions(+), 1 deletions(-)


# -------------- Then, fix the bug and commit the fix in one or more separate commits.
$ git commit -m "Fixed severe production problem"
# [hotfix-1.2.1 abbe5d6] Fixed severe production problem
# 5 files changed, 32 insertions(+), 17 deletions(-)


# -------------- Finishing and merge a hotfix branch
# the bugfix needs to be merged back into master and develop
# to safeguard that the bugfix is included in the next release as well.
# This is completely similar to how release branches are finished.

# Switched to branch 'master'
$ git checkout master
$ git merge --no-ff hotfix-1.2.1
$ git tag -a 1.2.1

# Switched to branch 'develop'
$ git checkout develop
$ git merge --no-ff hotfix-1.2.1

# The one exception to the rule here is that, when a release branch currently exists, the hotfix changes need to be merged into that release branch, instead of develop.
# Back-merging the bugfix into the release branch will eventually result in the bugfix being merged into develop too, when the release branch is finished.



# -------------- remove the temporary branch:
# Deleted branch hotfix-1.2.1 (was abbe5d6).
$ git branch -d hotfix-1.2.1
```






.
