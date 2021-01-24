/*
 * Find out the <a> tag contains an image and mark it.
 */

$(function() {
  const MARK = "img-hyperlink";
  $("a:has(img)").addClass(MARK);
});
