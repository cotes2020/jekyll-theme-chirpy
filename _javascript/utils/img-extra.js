/**
  Lazy load images (https://github.com/ApoorvSaxena/lozad.js)
  and popup when clicked (https://github.com/dimsemenov/Magnific-Popup)
*/

$(function() {

  const IMG_SCOPE = '#main > div.row:first-child > div:first-child';

  if ($(`${IMG_SCOPE} img`).length <= 0 ) {
    return;
  }

  /* lazy loading */

  const imgList = document.querySelectorAll(`${IMG_SCOPE} img[data-src]`);
  const observer = lozad(imgList);
  observer.observe();

  /* popup */

  $(`${IMG_SCOPE} p > img[data-src],${IMG_SCOPE} img[data-src].preview-img`).each(
    function() {
      let nextTag = $(this).next();
      const title = nextTag.prop('tagName') === 'EM' ? nextTag.text() : '';
      const src = $(this).attr('data-src'); // created by lozad.js

      $(this).wrap(`<a href="${src}" title="${title}" class="popup img-link"></a>`);
    }
  );

  $('.popup').magnificPopup({
    type: 'image',
    closeOnContentClick: true,
    showCloseBtn: false,
    zoom: {
      enabled: true,
      duration: 300,
      easing: 'ease-in-out'
    }
  });

  /* markup the image links */

});
