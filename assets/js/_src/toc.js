/*
 * Hide the empty ToC in posts.
 *
 * © 2019 Cotes Chung
 * MIT Licensed
 */

$(function() {
  if ($("#post-wrap .post-content h1").length == 0
      && $("#post-wrap .post-content h2").length == 0
      && $("#post-wrap .post-content h3").length == 0
      && $("#post-wrap .post-content h4").length == 0
      && $("#post-wrap .post-content h5").length == 0) {
    $("#toc-wrap").addClass("hidden");
  }
});