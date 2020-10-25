---
title: Release faster with Feature Toggles
date: 2019-02-17T14:59:31+01:00
author: Wolfgang Ofner
categories: [DevOps]
tags: [.net core, 'C#', Continous Deployment]
---
Nowadays, more and more teams do (finally) continuous integration and continuous deployment (CI/CD). Unfortunately, even if the development team was able to do CI/CD, the business department is often a limiting factor since they want to test the feature before it is deployed. So only teams without a limiting business department can do CI/CD? No, the solution to this problem are feature toggles. Today, I will show how you can CI/CD and still satisfy your bureaucratic business department.

## What are Feature Toggles?

Feature toggles or feature flags enable you to turn on or off your features. The main advantage is that you can deploy features which are not tested and hide them behind a feature toggle. Another usage of feature toggles can be to show different features to different users.  To understand this concept better, let me explain you to the difference between these two types of feature toggles in the next section

## Release Feature Toggles vs. Business Feature Toggle

Release feature toggles are used by the development team to deploy new features. These features can be untested or only needed at a specific time in the future. The advantage of feature toggles is that you can deploy all your features continuously and also can have them all in the same branch. So you don&#8217;t need several branches for future features. Release feature toggles stay in the application only for a short time and will be removed as soon as they are not needed anymore.

Business feature toggles, in contrast, are long living toggles and can make your application individual to a user role for example. You could have a free user where you have a limit set of features and a paid version with all features. The additional features for the paid version are hidden behind release feature toggles.

## Configuring Feature Toggles

There are three different approaches on how to configure your feature toggles. Choose the one which fits your system best.

### Compiled configuration

The configuration of the toggles is done during the deployment. This can be done by a simple hard-coded if statement within your code. The downside of this approach is, that a new release is needed if you want to change the configuration.

### Local configuration

Another approach is to put the configuration in a local file like the appsettings.json or the web.config file. The advantage is that no new release is needed to change the configuration. The downside is that every server needs its own config file.

### Centralized configuration

The last configuration is the centralized configuration. Here you have your configuration on a central location like a database or a file share. The advantages of this approach are that no release is needed to apply changes and that all systems can be managed centrally.

## Manually implementing Feature Toggles

In this simple example, I will write the configuration in the appsettings.json file of my .Net core web application and configure the Startup class to read the value. Afterward, I will change the title of my page, depending on the value of the toggle. I am calling my toggle in the appsettings.json FeatureToggle and set the IsEnabled property to true.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/02/Enable-the-toggle-in-the-appsettings.jpg"><img loading="lazy" src="/assets/img/posts/2019/02/Enable-the-toggle-in-the-appsettings.jpg" alt="Enable the toggle in the appsettings" /></a>
  
  <p>
    Enable the toggle in the appsettings.json file
  </p>
</div>

Next, I enable Options in the Startup class to read the settings individually and then read my value into the MyFeatureToggleClass.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/02/Set-up-reading-the-appsettings-file.jpg"><img loading="lazy" src="/assets/img/posts/2019/02/Set-up-reading-the-appsettings-file.jpg" alt="Set up reading the appsettings file" /></a>
  
  <p>
    Set up reading the appsettings file
  </p>
</div>

I put the MyFeatureToggle class in a new folder called Toggles and its only property is a bool indicating whether it is enabled.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/02/The-feature-toggle-class.jpg"><img loading="lazy" src="/assets/img/posts/2019/02/The-feature-toggle-class.jpg" alt="The feature toggle class" /></a>
  
  <p>
    The feature toggle class
  </p>
</div>

As a simple test if my set up is working, I read the value and if my toggle is enabled, I set the view title to &#8220;Toggle is enabled&#8221;. Otherwise, I display &#8220;Toggle is disabled&#8221;.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/02/Setting-the-title-depending-on-the-value-of-the-toggle.jpg"><img loading="lazy" src="/assets/img/posts/2019/02/Setting-the-title-depending-on-the-value-of-the-toggle.jpg" alt="Setting the title depending on the value of the toggle" /></a>
  
  <p>
    Setting the title depending on the value of the toggle
  </p>
</div>

Start the application and click the Privacy menu item. Now you will see that the toggle is enabled.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/02/Test-the-enabled-feature-toggle.jpg"><img loading="lazy" src="/assets/img/posts/2019/02/Test-the-enabled-feature-toggle.jpg" alt="Test the enabled feature toggle" width="657" height="220"  /></a>
  
  <p>
    Test the enabled feature toggle
  </p>
</div>

### Limitations of the manual approach

A problem with the manual implementation is that you have magic strings in your code (in your Startup class) which can lead to runtime exceptions if you have a typo in it. Another exception can occur if you delete a feature toggle but forget to remove the check inside the code. Therefore, it is a good idea to take a look at one of the many open source libraries.

## Libraries for Feature Toggles

There is a number of open source libraries for feature toggles such as:

  * <a href="https://github.com/benaston/NFeature" target="_blank" rel="noopener">NFeature</a>
  * <a href="https://github.com/mexx/FeatureSwitcher" target="_blank" rel="noopener">FeatureSwitcher</a>
  * <a href="https://github.com/jason-roberts/FeatureToggle" target="_blank" rel="noopener">FeatureToggle</a>

In the following section, I will introduce you to the FeatureToggle library.

### FeatureToggle

To install the FeatureToggle library, you only have to download the NuGet package. The advantage of using the library is that you don&#8217;t have any magic strings in your code and also no default values which might lead to unexpected behavior if no setting for the feature toggle is available.

The FeatureToggle library provides the IFeatureToggle interface which has to be implemented by your feature toggle class. The biggest advantage of the library is that it comes with a couple of built-in feature toggles like:

  * AlwaysOnFeatureToggle / AlwaysOffFeatureToggle
  * EnabledOnOrAfterDateFeatureToggle / EnabledOnOrBeforeDateFeatureToggle
  * EnabledBetweenDatesFeatureToggle
  * EnabledOnDaysOfWeekFeatureToggle
  * RandomFeatureToggle

#### AlwaysOnFeatureToggle / AlwaysOffFeatureToggle

This is the simplest feature toggle available. It is either always on or off, depending on its name.

#### EnabledOnOrAfterDateFeatureToggle / EnabledOnOrBeforeDateFeatureToggle

This toggle helps you to activate or deactivate a feature before or after a certain date. For example, this can be used to enable a sale coupon after a certain date.

#### EnabledBetweenDatesFeatureToggle

The enable between date toggle is similar to the before mentioned one. It can be used to enable a feature only during a specific time span. For example for a sale between Christmas and New Years Eve.

#### EnabledOnDaysOfWeekFeatureToggle

With this toggle, you can Enable your feature on one or several days of the week. For example, you can have a sale code which is only available from Friday until Sunday.

#### RandomFeatureToggle

As the name already suggests, this toggle is enabled randomly. This is great to do A/B testing because it shows a feature only to a random group of people. For example, this group sees a new search. Measure the conversation rate and compare it to the old search feature and so you can see if the new search performs better than the old one or not.

## Conclusion

In this post, I showed how you can use Feature Toggles to continuously deploy your new features, even without activating them for the user. Additionally, I showed how to implement your own feature toggles and how to use the open source library FeatureToggle. You can find the code for the manually implemented feature toggles on <a href="https://github.com/WolfgangOfner/Feature-Toggles" target="_blank" rel="noopener">GitHub</a>.

If you want to learn more about Feature Toggles, I can recommend the Pluralsight course &#8220;<a href="https://app.pluralsight.com/library/courses/dotnet-featuretoggle-implementing/table-of-contents" target="_blank" rel="noopener">Implementing Feature Toggles in .NET with FeatureToggle</a>&#8221; by Jason Roberts.