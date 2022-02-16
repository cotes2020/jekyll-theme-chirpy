/*
 * Listener for theme mode toggle
 */
$(function() {
  $(".mode-toggle").click((e) => {
    const $target = $(e.target);
    let $btn = ($target.prop("tagName") === "button".toUpperCase() ?
      $target : $target.parent());

    $btn.blur(); // remove the clicking outline
    flipMode();
  });
});
