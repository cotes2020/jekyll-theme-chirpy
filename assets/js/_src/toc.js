/*
 * Hide the empty ToC in posts.
 *
 * © 2019 Cotes Chung
 * MIT Licensed
 */

$(function() {
  // Hide ToC title if there is no head
  if ($("#toc-wrap>nav#toc>ul>li").length == 0) {
    $("#toc-wrap>h3").addClass("hidden");
  }
});