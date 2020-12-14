/*
 * Create a more beautiful checkbox
 */

$(function() {
  /* hide browser default checkbox */
  $("input[type=checkbox]").addClass("unloaded");
  /* create checked checkbox */
  $("input[type=checkbox][checked]").before("<span checked></span>");
  /* create normal checkbox */
  $("input[type=checkbox]:not([checked])").before("<span></span>");
});
