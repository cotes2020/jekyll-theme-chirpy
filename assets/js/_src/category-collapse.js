/*
 * Tab 'Categories' expand/close effect.
 * v2.0
 * https://github.com/cotes2020/jekyll-theme-chirpy
 * © 2018-2019 Cotes Chung
 * MIT License
 */

$(function() {
  var child_prefix = "l_";
  var parent_prefix = "h_";

  // close up top-category
  $(".collapse").on("hide.bs.collapse", function() { // Bootstrap collapse events.
    var parent_id = parent_prefix + $(this).attr('id').substring(child_prefix.length);
    if (parent_id) {
      $("#" + parent_id + " .far.fa-folder-open").attr("class", "far fa-folder fa-fw");
      $("#" + parent_id + " i.fas").addClass("rotate");
      $("#" + parent_id).removeClass("hide-border-bottom");
    }
  });

  // expand the top category
  $(".collapse").on("show.bs.collapse", function() {
    var parent_id = parent_prefix + $(this).attr('id').substring(child_prefix.length);
    if (parent_id) {
      $("#" + parent_id + " .far.fa-folder").attr("class", "far fa-folder-open fa-fw");
      $("#" + parent_id + " i.fas").removeClass("rotate");
      $("#" + parent_id).addClass("hide-border-bottom");
    }
  });

});