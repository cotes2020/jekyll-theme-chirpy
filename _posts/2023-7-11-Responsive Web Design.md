---
title: Web Design 练习
date: 2023-7-11
categories: [Web, Responsive Web Design]
tags: [Web, front end, Responsive Web Design]     # TAG names should always be lowercase
---

### Build a Product Landing Page

> Objective: Build an app that is functionally similar to https://product-landing-page.freecodecamp.rocks



### Code Pard
HTML Code
```
<!DOCTYPE html>
<html lang="en-US">
  <head>
    <link ref="stylesheet" href="styles.css">
  </head>
  <body>
    <div id="product-landing-page">
      <header id="header">
        <div id="logo">
          <a href="#" class="logo-link">
            <img id="header-img" src="https://cdn.freecodecamp.org/testable-projects-fcc/images/product-landing-page-logo.png" alt="original trombones logo">
          </a>
        </div>
        <nav id="nav-bar">
          <ul>
            <li>
              <a class="nav-link" href="#features">Features</a>
            </li>
            <li>
              <a class="nav-link" href="#how-it-works">How It Works</a>
            </li>
            <li>
              <a class="nav-link" href="#pricing">Pricing</a>
            </li>
          </ul>
        </nav>
      </header>
      <section id="hero">
        <h2>Home-made masterpieces</h2>
        <form id="form" action="https://www.freecodecamp.com/email-submit">
          <input name="email" id="email" type="email" placeholder="Enter your email address" required="">
          <input id="submit" type="submit" value="Get Started" class="btn">
        </form>
      </section>
      <div class="container">
        <section id="features">
          <div class="grid">
            <div class="icon"><i class="fa fa-3x fa-fire"></i></div>
            <div class="desc">
              <h2>Premium Materials</h2>
              <p>
                Our trombones use the shiniest brass which is sourced locally.
                This will increase the longevity of your purchase.
              </p>
            </div>
          </div>
          <div class="grid">
            <div class="icon"><i class="fa fa-3x fa-truck"></i></div>
            <div class="desc">
              <h2>Fast Shipping</h2>
              <p>
                We make sure you recieve your trombone as soon as we have
                finished making it. We also provide free returns if you are not
                satisfied.
              </p>
            </div>
          </div>
          <div class="grid">
            <div class="icon">
              <i class="fa fa-3x fa-battery-full" aria-hidden="true"></i>
            </div>
            <div class="desc">
              <h2>Quality Assurance</h2>
              <p>
                For every purchase you make, we will ensure there are no damages
                or faults and we will check and test the pitch of your
                instrument.
              </p>
            </div>
          </div>
        </section>
        <section id="how-it-works">
          <iframe id="video" src="https://www.youtube-nocookie.com/embed/y8Yv4pnO7qc?rel=0&amp;controls=0&amp;showinfo=0" allowfullscreen="" height="315" frameborder="0"></iframe>
        </section>
        <section id="pricing">
          <div class="product" id="tenor">
            <div class="level">Tenor Trombone</div>
            <h2>$600</h2>
            <ol>
              <li>Lorem ipsum.</li>
              <li>Lorem ipsum.</li>
              <li>Lorem ipsum dolor.</li>
              <li>Lorem ipsum.</li>
            </ol>
            <button class="btn">Select</button>
          </div>
          <div class="product" id="bass">
            <div class="level">Bass Trombone</div>
            <h2>$900</h2>
            <ol>
              <li>Lorem ipsum.</li>
              <li>Lorem ipsum.</li>
              <li>Lorem ipsum dolor.</li>
              <li>Lorem ipsum.</li>
            </ol>
            <button class="btn">Select</button>
          </div>
          <div class="product" id="valve">
            <div class="level">Valve Trombone</div>
            <h2>$1200</h2>
            <ol>
              <li>Plays similar to a Trumpet</li>
              <li>Great for Jazz Bands</li>
              <li>Lorem ipsum dolor.</li>
              <li>Lorem ipsum.</li>
            </ol>
            <button class="btn">Select</button>
          </div>
        </section>
        <footer>
          <ul>
            <li><a href="#">Privacy</a></li>
            <li><a href="#">Terms</a></li>
            <li><a href="#">Contact</a></li>
          </ul>
          <span>Copyright 2016, Original Trombones</span>
        </footer>
      </div>
    </div>
  </body>
</html>
```

CSS Code
```

```