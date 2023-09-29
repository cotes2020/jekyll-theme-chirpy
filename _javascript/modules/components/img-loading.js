/**
 * Setting up image lazy loading and LQIP switching
 */

const ATTR_DATA_SRC = 'data-src';
const ATTR_DATA_LQIP = 'data-lqip';
const C_SHIMMER = 'shimmer';
const C_BLUR = 'blur';

function handleImage() {
  const $img = $(this);

  if (this.hasAttribute(ATTR_DATA_LQIP) && this.complete) {
    $img.parent().removeClass(C_BLUR);
  } else {
    $img.parent().removeClass(C_SHIMMER);
  }
}

/* Switch LQIP with real image url */
function switchLQIP(img) {
  // Sometimes loaded from cache without 'data-src'
  if (img.hasAttribute(ATTR_DATA_SRC)) {
    const $img = $(img);
    const dataSrc = $img.attr(ATTR_DATA_SRC);
    $img.attr('src', encodeURI(dataSrc));
  }
}

export function loadImg() {
  const $images = $('article img');

  if ($images.length) {
    $images.on('load', handleImage);
  }

  /* Images loaded from the browser cache do not trigger the 'load' event */
  $('article img[loading="lazy"]').each(function () {
    if (this.complete) {
      $(this).parent().removeClass(C_SHIMMER);
    }
  });

  const $lqips = $(`article img[${ATTR_DATA_LQIP}="true"]`);

  if ($lqips.length) {
    const isHome = $('#post-list').length > 0;

    $lqips.each(function () {
      if (isHome) {
        // JavaScript runs so fast that LQIPs in home page will never be detected
        // Delay 50ms to ensure LQIPs visibility
        setTimeout(() => {
          switchLQIP(this);
        }, 50);
      } else {
        switchLQIP(this);
      }
    });
  }
}
