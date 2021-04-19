/*
 * Find the image links and mark them
 */

$(function() {
  const MARK = "img-link";
  $("#main a").has("img").addClass(MARK);
});
