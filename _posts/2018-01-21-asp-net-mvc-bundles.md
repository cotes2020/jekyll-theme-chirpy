---
title: ASP.NET MVC Bundles
date: 2018-01-21T00:19:37+01:00
author: Wolfgang Ofner
categories: [ASP.NET]
tags: [CSS, Javascript, minification, MVC, Optimization]
---
The Bundles feature is built-in into the MVC framework and helps to organize and optimize CSS and Javascript files. In this post, I will show what bundling is, how to use it and what effects it has on the performance of your application.

## Setting up the project

To show how bundles word, I create a new empty ASP.NET MVC project with the MVC folders. Then I install the following NuGet packages:

  * Bootstrap
  * <span class="fontstyle0">jQuery </span>
  *  <span class="fontstyle0">jQuery.Validation</span>
  * <span class="fontstyle0">Microsoft.jQuery.Unobtrusive.Validation</span>
  * <span class="fontstyle0">Microsoft.jQuery.Unobtrusive.Ajax</span>

These CSS and Javascript libraries are often used, so they represent a real-world example. Then I create a simple view which uses at least one of the features of each package so that there are several files which need to be loaded.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/01/Adding-CSS-and-Javascript-files.jpg"><img loading="lazy" src="/assets/img/posts/2018/01/Adding-CSS-and-Javascript-files.jpg" alt="Adding CSS and Javascript files" /></a>
  
  <p>
    Adding CSS and Javascript files
  </p>
</div>

## Analyzing the network load

You can analyze the network load with every browser by pressing the F12 key. This opens the dev tools which shows some useful information for development and debugging. Switch to the Network tab and disable Caching. Then reload the website. This lists all files requested and also displays a summary of the number of files transferred and the number of bytes.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/01/Network-tab-in-the-Chrome-DevTools.jpg"><img loading="lazy" src="/assets/img/posts/2018/01/Network-tab-in-the-Chrome-DevTools.jpg" alt="Network tab in the Chrome DevTools" /></a>
  
  <p>
    Network tab in the Chrome DevTools
  </p>
</div>

&nbsp;

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/01/Summary-of-the-transferred-data.jpg"><img loading="lazy" src="/assets/img/posts/2018/01/Summary-of-the-transferred-data.jpg" alt="Summary of the transferred data" /></a>
  
  <p>
    Summary of the requests and transferred data
  </p>
</div>

As you can see there were quite some files transferred for such a simple page. In total nine requests were sent and 649 KB of data. This is a lot of data for such a simple page and could be a problem when the page grows and hosts more content. The solution for too many requests and too many sent data is bundling.

## What is Bundling?

In the bundling process, the ASP.NET MVC framework minifies Javascript and CSS files to reduce the file size and therefore the amount of data which needs to be sent to the browser. The reduction of the file size is achieved by removing white spaces and shortening variable and method names. Often these minified files are already downloaded when installing a NuGet package, for example, while installing jQuery, the Packet Manager also installed the minified version. You can detect such a file at the ending .min.js. Due to the minification, these files are hardly human readable.

## Setting up Bundles

The first step to use Bundles is installing the Microsoft.AspNet.Web.Optimization packages from NuGet. Then add the BundleConfig class to the App_Start folder. You don&#8217;t have to do this if you use the MVC template when creating a new project. Then Visual Studio automatically creates the class and installs the NuGet package for you.

The BundleConfig class has one static method,  <span class="fontstyle0">RegisterBundles</span> which takes a BundleCollection as parameter. Add your CSS files by adding a new StyleBundle to the BundleCollection and the Javascript files by adding a new ScriptBundle to the BundleCollection.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/01/Adding-CSS-and-Javascript-Bundles-to-the-BundleConfig-class.jpg"><img loading="lazy"  title="Adding CSS and Javascript Bundles to the BundleConfig class" src="/assets/img/posts/2018/01/Adding-CSS-and-Javascript-Bundles-to-the-BundleConfig-class.jpg" alt="Adding CSS and Javascript Bundles to the BundleConfig class" /></a>
  
  <p>
    Adding CSS and Javascript Bundles to the BundleConfig class
  </p>
</div>

You can add files with the full name or you could wildcards if you want to use all files from a folder, for example, you could add all Javascript files from the Scripts folder with &#8220;~/Scripts/*.js&#8221;. Note that I used the version variable for jQuery. This means that every jQuery version in the Scripts folder gets included. The advantage is that I can update jQuery and don&#8217;t have to think about adding the version number in the Bundle. The downside is that if I have several jQuery files in the folder, all get loaded.

If one file depends on the other one, make sure to add them in the right order. For example jQuery.validate needs jQuery, therefore the jQuery file needs to be added first.

## Using Bundles in ASP.NET MVC

After creating the Bundles, I have to add the RegisterBundle method in the Global.asx file to make sure it is called at the start of the application. To register the RegisterBunle method in the Global.asx file add the following line of code: <span class="fontstyle0">BundleConfig.RegisterBundles(BundleTable.Bundles);</span>

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/01/Registering-the-Bundles-in-the-Global.asax-file.jpg"><img loading="lazy" src="/assets/img/posts/2018/01/Registering-the-Bundles-in-the-Global.asax-file.jpg" alt="Registering the Bundles in the Global.asax file" /></a>
  
  <p>
    Registering the Bundles in the Global.asax file
  </p>
</div>

The next step is to add the System.Web.Optimization namespace to the web.config viel in the Views folder.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/01/Adding-the-System.Web_.Optimization-namespace-to-the-web.config-file.jpg"><img loading="lazy" src="/assets/img/posts/2018/01/Adding-the-System.Web_.Optimization-namespace-to-the-web.config-file.jpg" alt="Adding the System.Web.Optimization namespace to the web.config file" /></a>
  
  <p>
    Adding the System.Web.Optimization namespace to the web.config file
  </p>
</div>

The last step is adding the Bundles to your layout view and enabling CSS and Javascript optimization in the web.config file. To do that use Styles.Render and Scripts.Render with the name of the Bundle you provided while creating it.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/01/Adding-the-Bundles-to-the-layout-view.jpg"><img loading="lazy" src="/assets/img/posts/2018/01/Adding-the-Bundles-to-the-layout-view.jpg" alt="Adding the Bundles to the layout view" /></a>
  
  <p>
    Adding the Bundles to the layout view
  </p>
</div>

Then set the value of the debug attribute in the compilation tag in the web.config to false. If this is set to true, the browser requests all the files with separate requests and doesn&#8217;t use bundling.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/01/Setting-debug-to-false-in-the-web.config.jpg"><img loading="lazy" src="/assets/img/posts/2018/01/Setting-debug-to-false-in-the-web.config.jpg" alt="Setting debug to false in the web.config" /></a>
  
  <p>
    Setting debug to false in the web.config
  </p>
</div>

In theory, fewer files get requested and the transferred data size is smaller too now. To test this I start the application and open the DevTools again. Then I reload the page.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/01/Network-tab-in-the-Chrome-DevTools-after-Bundling.jpg"><img loading="lazy" src="/assets/img/posts/2018/01/Network-tab-in-the-Chrome-DevTools-after-Bundling.jpg" alt="Network tab in the Chrome DevTools after Bundling" /></a>
  
  <p>
    Network tab in the Chrome DevTools after bundling
  </p>
</div>

As you can see only 3 requests and only 272 KB were sent. The request was finished after 53 ms instead of 127 ms without bundling.

## Conclusion

In this post, I showed how to optimize your CSS and Javascript file by minimizing them using the ASP.NET MVS Bundling feature. With this feature, the number of requests, transferred data and time to finish got significantly decreased.

For more details about how bundles work, I highly recommend the books <a href="http://amzn.to/2mgRbTy" target="_blank" rel="noopener">Pro ASP.NET MVC 5</a> and <a href="http://amzn.to/2mfQ0nA" target="_blank" rel="noopener">Pro ASP.NET MVC 5 Plattform</a>.

You can find the source code on <a href="https://github.com/WolfgangOfner/MVC-Bundles" target="_blank" rel="noopener">GitHub</a>.