/*
 * Document-reday functions for '#mode-toggle-wrapper'
 */
$(function() {
  $("#mode-toggle-wrapper").keyup((e) => {
    if(e.keyCode === 13) {
      flipMode();
    }
  });
});
