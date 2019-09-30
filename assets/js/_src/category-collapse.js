/*
 * Tab 'Categories' expand/close effect.
 * © 2018-2019 Cotes Chung
 * MIT License
 */

$(function() {
  var child_prefix = "l_";
  var parent_prefix = "h_";

  // close up Top
  $(".collapse").on("hide.bs.collapse", function() { // Bootstrap collapse events.
    var parent_id = parent_prefix + $(this).attr('id').substring(child_prefix.length);
    if (parent_id) {
      $("#" + parent_id + " .far.fa-folder-open").attr("class", "far fa-folder fa-fw");
      $("#" + parent_id + " i.fas.fa-angle-up").addClass("flip");
      $("#" + parent_id).removeClass("hide-border-bottom");
    }
  });

  // expand Top Category
  $(".collapse").on("show.bs.collapse", function() {
    var parent_id = parent_prefix + $(this).attr('id').substring(child_prefix.length);
    if (parent_id) {
      $("#" + parent_id + " .far.fa-folder").attr("class", "far fa-folder-open fa-fw");
      $("#" + parent_id + " i.fas.fa-angle-up").removeClass("flip");
      $("#" + parent_id).addClass("hide-border-bottom");
    }
  });

});