/*
 * Find out the <a> tag contains an image and mark it.
 *
 * v2.5.1
 * https://github.com/cotes2020/jekyll-theme-chirpy
 * © 2020 Cotes Chung
 * MIT Licensed
 */

$(function() {

  var MARK="img-hyperlink";

  $("a:has(img)").addClass(MARK);
  
});
