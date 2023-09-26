/**
 * Setting up image lazy loading and LQIP switching
 */

export function loadImg() {
  const $images = $('main img[loading="lazy"]');
  const $lqip = $('main img[data-lqip="true"]');

  if ($images.length > 0) {
    $images.on('load', function () {
      /* Stop shimmer when image loaded */
      $(this).parent().removeClass('shimmer');
    });

    $images.each(function () {
      /* Images loaded from the browser cache do not trigger the 'load' event */
      if ($(this).prop('complete')) {
        $(this).parent().removeClass('shimmer');
      }
    });
  }

  if ($lqip.length > 0) {
    $lqip.each(function () {
      /* Switch LQIP with real image url */
      const dataSrc = $(this).attr('data-src');
      $(this).attr('src', encodeURI(dataSrc));
      $(this).removeAttr('data-src data-lqip');
    });
  }
}
