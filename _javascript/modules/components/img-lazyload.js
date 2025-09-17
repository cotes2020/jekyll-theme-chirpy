/**
 * Set up image lazy-load
 */

function stopShimmer($node) {
  $node.parent().removeClass('shimmer');
}

export function imgLazy() {
  const $images = $('#core-wrapper img[data-src]');

  if ($images.length <= 0) {
    return;
  }

  /* Stop shimmer when image loaded */
  document.addEventListener('lazyloaded', function (e) {
    stopShimmer($(e.target));
  });

  /* Stop shimmer from cached images */
  $images.each(function () {
    if ($(this).hasClass('ls-is-cached')) {
      stopShimmer($(this));
    }
  });
}
