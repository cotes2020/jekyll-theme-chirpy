/**
 * Add listener for theme mode toggle
 */
const $toggleElem = $('.mode-toggle');

export function modeWatcher() {
  if ($toggleElem.length === 0) {
    return;
  }

  $toggleElem.off().on('click', (e) => {
    const $target = $(e.target);
    let $btn =
      $target.prop('tagName') === 'button'.toUpperCase()
        ? $target
        : $target.parent();

    modeToggle.flipMode(); // modeToggle: `_includes/mode-toggle.html`
    $btn.trigger('blur'); // remove the clicking outline
  });
}
