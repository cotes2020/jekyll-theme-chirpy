/**
 * Setting up image lazy loading and LQIP switching
 */

const ATTR_DATA_SRC = 'data-src';
const ATTR_DATA_LQIP = 'data-lqip';

const cover = {
  SHIMMER: 'shimmer',
  BLUR: 'blur'
};

function removeCover(clzss) {
  $(this).parent().removeClass(clzss);
}

function handleImage() {
  if (!this.complete) {
    return;
  }

  if (this.hasAttribute(ATTR_DATA_LQIP)) {
    removeCover.call(this, cover.BLUR);
  } else {
    removeCover.call(this, cover.SHIMMER);
  }
}

/**
 * Switches the LQIP with the real image URL.
 */
function switchLQIP() {
  const $img = $(this);
  const src = $img.attr(ATTR_DATA_SRC);

  $img.attr('src', encodeURI(src));
  $img.removeAttr(ATTR_DATA_SRC);
}

export function loadImg() {
  const $images = $('article img');

  if ($images.length) {
    $images.on('load', handleImage);
  }

  // Images loaded from the browser cache do not trigger the 'load' event
  $('article img[loading="lazy"]').each(function () {
    if (this.complete) {
      removeCover.call(this, cover.SHIMMER);
    }
  });

  // LQIPs set by the data URI or WebP will not trigger the 'load' event,
  // so manually convert the URI to the URL of a high-resolution image.
  const $lqips = $(`article img[${ATTR_DATA_LQIP}="true"]`);

  if ($lqips.length) {
    $lqips.each(switchLQIP);
  }
}
