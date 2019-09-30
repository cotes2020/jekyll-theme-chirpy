/*
 * Hide Header on scroll down
 * © 2018-2019 Cotes Chung
 * MIT License
 */
$(function() {

  var didScroll;
  var lastScrollTop = 0;
  var delta = 5;
  var topbarHeight = $('#topbar').outerHeight();

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
      $('#topbar').removeClass('topbar-down').addClass('topbar-up');

      if ( $('#toc-wrap').length > 0) {
        $('#toc-wrap').removeClass('topbar-down');
      }

      if ( $('.panel-group').length > 0) {
        $('.panel-group').removeClass('topbar-down');
      }

      if ($('#search-input').is(':focus')) {
        $('#search-input').blur(); // remove focus
      }

    } else {
      // Scroll Up
      if (st + $(window).height() < $(document).height()) {
        $('#topbar').removeClass('topbar-up').addClass('topbar-down');
        if ( $('#toc-wrap').length > 0) {
          $('#toc-wrap').addClass('topbar-down');
        }
        if ( $('.panel-group').length > 0) {
          $('.panel-group').addClass('topbar-down');
        }
      }
    }

    lastScrollTop = st;
  }
});