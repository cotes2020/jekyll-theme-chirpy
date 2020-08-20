/*
 * Tab 'Categories' expand/close effect.
 * v2.0
 * https://github.com/cotes2020/jekyll-theme-chirpy
 * © 2018-2019 Cotes Chung
 * MIT License
 */

$(function() {
  var childPrefix = "l_";
  var parentPrefix = "h_";

  /* close up top-category */
  $(".collapse").on("hide.bs.collapse", function() { /* Bootstrap collapse events. */
    var parentId = parentPrefix + $(this).attr("id").substring(childPrefix.length);
    if (parentId) {
      $("#" + parentId + " .far.fa-folder-open").attr("class", "far fa-folder fa-fw");
      $("#" + parentId + " i.fas").addClass("rotate");
      $("#" + parentId).removeClass("hide-border-bottom");
    }
  });

  /* expand the top category */
  $(".collapse").on("show.bs.collapse", function() {
    var parentId = parentPrefix + $(this).attr("id").substring(childPrefix.length);
    if (parentId) {
      $("#" + parentId + " .far.fa-folder").attr("class", "far fa-folder-open fa-fw");
      $("#" + parentId + " i.fas").removeClass("rotate");
      $("#" + parentId).addClass("hide-border-bottom");
    }
  });

});