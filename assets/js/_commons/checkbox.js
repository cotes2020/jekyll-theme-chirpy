/*
 * Create a more beautiful checkbox
 */

$(function() {
  /* hide bullet of checkbox item */
  $("li.task-list-item:has(input)").attr("hide-bullet", "");
  /* create checked checkbox */
  $("input[type=checkbox][checked=checked]").before("<span checked></span>");
  /* create normal checkbox */
  $("input[type=checkbox]:not([checked=checked])").before("<span></span>");
});
