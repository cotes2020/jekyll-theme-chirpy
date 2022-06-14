/*
 * Create a more beautiful checkbox
 */

$(function () {
  /* hide browser default checkbox */
  $("input[type=checkbox]").addClass("unloaded");
  /* create checked checkbox */
  $("input[type=checkbox][checked]").before("<i class=\"bi check2-circle checked\"></i>");
  /* create normal checkbox */
  $("input[type=checkbox]:not([checked])").before("<i class=\"bi bi-circle\"></i>");
});
