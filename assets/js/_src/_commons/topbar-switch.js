/*
 * Hide Header on scroll down
 * v2.0
 * https://github.com/cotes2020/jekyll-theme-chirpy
 * © 2018-2019 Cotes Chung
 * MIT License
 */
$(function() {

  var didScroll;
  var lastScrollTop = 0;
  var delta = 5;
  var topbarHeight = $('#topbar-wrapper').outerHeight();

  $(window).scroll(function(event) {
    if ($("#topbar-title").is(":hidden")) { // Not in small screens
      didScroll = true;
    }
  });

  setInterval(function() {
    if (didScroll) {
      hasScrolled();
      didScroll = false;
    }
  }, 250);

  function hasScrolled() {
    var st = $(this).scrollTop();

    // Make sure they scroll more than delta
    if (Math.abs(lastScrollTop - st) <= delta)
      return;

    if (st > lastScrollTop && st > topbarHeight) {
      // Scroll Down
      $('#topbar-wrapper').removeClass('topbar-down').addClass('topbar-up');

      if ( $('#toc-wrapper').length > 0) {
        $('#toc-wrapper').removeClass('topbar-down');
      }

      if ( $('.access').length > 0) {
        $('.access').removeClass('topbar-down');
      }

      if ($('#search-input').is(':focus')) {
        $('#search-input').blur(); // remove focus
      }

    } else {
      // Scroll Up
      if (st + $(window).height() < $(document).height()) {
        $('#topbar-wrapper').removeClass('topbar-up').addClass('topbar-down');
        if ( $('#toc-wrapper').length > 0) {
          $('#toc-wrapper').addClass('topbar-down');
        }
        if ( $('.access').length > 0) {
          $('.access').addClass('topbar-down');
        }
      }
    }

    lastScrollTop = st;
  }
});