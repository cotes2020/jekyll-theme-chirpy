
# Client side Web API

[toc]

---

## basic

**Client-side JavaScript** has many APIs available
- not part of the JavaScript language itself
- built on top of the core JavaScript language, providing extra superpowers to use in JavaScript code.
- two categories:
  - Browser APIs
    - built into web browser and are able to expose data from the browser and surrounding computer environment and do useful complex things with it.
    - For example, the Web Audio API provides JavaScript constructs for manipulating audio in the browser — taking an audio track, altering its volume, applying effects to it, etc. In the background, the browser is actually using some complex lower-level code (e.g. C++ or Rust) to do the actual audio processing. But again, this complexity is abstracted away from you by the API
  - Third-party APIs
    - are not built into the browser by default, and have to retrieve their code and information from somewhere on the Web.
    - For example, the Twitter API allows you to do things like displaying your latest tweets on your website. It provides a special set of constructs you can use to query the Twitter service and return specific information.


**Relationship**
- **JavaScript**
  - A high-level scripting language built into browsers
  - allows to implement functionality on web pages/apps.
  - also available in other programming environments, like Node.
- **Browser APIs**
  - constructs built into the browser that sits on top of the JavaScript language
  - allows to implement functionality more easily.
- **Third-party APIs**
  - constructs built into third-party platforms (e.g. Twitter, Facebook)
  - allow to use some of those platform's functionality in your own web pages (for example, display your latest Tweets on your web page).
- **JavaScript libraries**
  - JavaScript files containing custom functions that can attach to your web page to speed up or enable writing common functionality.
  - Examples include jQuery, Mootools and React.
- **JavaScript frameworks**
  - The next step up from libraries, JavaScript frameworks (e.g. Angular and Ember) tend to be packages of HTML, CSS, JavaScript, and other technologies that install and then use to write an entire web application from scratch.
  - The key difference between a library and a framework is “Inversion of Control”.
    - calling method from library, the developer is in control.
    - With a framework, the control is inverted: the framework calls the developer's code.



What can APIs do?

**Common browser APIs**
- **manipulating documents** loaded into the browser.
  - The most obvious example is the `DOM (Document Object Model) API`, manipulate HTML and CSS — creating, removing and changing HTML, dynamically applying new styles to your page, etc. Every time you see a popup window appear on a page or some new content displayed.
- **fetch data from the server** to update small sections of a webpage on their own.
  - huge impact on the performance and behaviour of sites — if you just need to update a stock listing or list of available new stories, doing it instantly without having to reload the whole entire page from the server can make the site or app feel much more responsive and "snappy".
  - APIs that make this possible include `XMLHttpRequest` and the `Fetch API`. You may also come across the term Ajax, which describes this technique.
- **drawing and manipulating graphics**
  - the most popular ones are `Canvas` and `WebGL`, programmatically update the pixel data contained in an HTML `<canvas>` element to create 2D and 3D scenes.
  - For example, you might draw shapes such as rectangles or circles, import an image onto the canvas, and apply a filter to it such as sepia or grayscale using the Canvas API, or create a complex 3D scene with lighting and textures using WebGL.
  - Such APIs are often combined with APIs for creating animation loops (such as `window.requestAnimationFrame()`) and others to make constantly updating scenes like cartoons and games.
- **Audio and Video APIs**
  - like `HTMLMediaElement`, the `Web Audio API`, and `WebRTC` do things with multimedia
  - such as creating custom UI controls for playing audio and video, displaying text tracks like captions and subtitles along with your videos, grabbing video from your web camera to be manipulated via a canvas or displayed on someone else's computer in a web conference, or adding effects to audio tracks (such as gain, distortion, panning, etc).
- **Device APIs**
  - basically APIs for manipulating and retrieving data from modern device hardware in a way that is useful for web apps.
  - Examples, telling user that a useful update is available on a web app via system notifications (Notifications API) or vibration hardware (Vibration API).
- **Client-side storage APIs** are becoming a lot more widespread in web browsers
  - the ability to store data on the client-side is very useful if you want to create an app that will save its state between page loads, and perhaps even work when the device is offline.
  - There are a number of options available, e.g. simple name/value storage with the `Web Storage API`, and more complex tabular data storage with the `IndexedDB API`.

**Common third-party APIs**
- twitter API, Facebook API, Google Maps API...


### How do APIs work?

**based on objects**
