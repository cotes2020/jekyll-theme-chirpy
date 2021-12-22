/*
 * Hide Header on scroll down
 */

$(function() {
<<<<<<< HEAD

  const topbarWrapper = $("#topbar-wrapper");
  const toc = $("#toc-wrapper");
  const access = $(".access");
  const searchInput = $("#search-input");
=======
  const $topbarWrapper = $("#topbar-wrapper");
  const $panel = $("#panel-wrapper");
  const $searchInput = $("#search-input");

  const CLASS_TOPBAR_UP = "topbar-up";
  const CLASS_TOPBAR_DOWN = "topbar-down";
  const ATTR_TOC_SCROLLING_UP = "toc-scrolling-up"; // topbar locked
>>>>>>> ebb3dc940c22d864dc41a16f1d84c1a0c0a003ba

  let didScroll;
  let lastScrollTop = 0;

<<<<<<< HEAD
  const delta = 5;
  const topbarHeight = topbarWrapper.outerHeight();

  function hasScrolled() {
    var st = $(this).scrollTop();
=======
  const delta = $topbarWrapper.outerHeight();
  const topbarHeight = $topbarWrapper.outerHeight();

  function hasScrolled() {
    let st = $(this).scrollTop();
>>>>>>> ebb3dc940c22d864dc41a16f1d84c1a0c0a003ba

    /* Make sure they scroll more than delta */
    if (Math.abs(lastScrollTop - st) <= delta) {
      return;
    }

<<<<<<< HEAD
    if (st > lastScrollTop && st > topbarHeight) {
      /* Scroll Down */
      topbarWrapper.removeClass("topbar-down").addClass("topbar-up");

      if (toc.length > 0) {
        toc.removeClass("topbar-down");
      }

      if (access.length > 0) {
        access.removeClass("topbar-down");
      }

      if (searchInput.is(":focus")) {
        searchInput.blur(); /* remove focus */
      }

    } else if (st + $(window).height() < $(document).height()) {
      /* Scroll Up */
      topbarWrapper.removeClass("topbar-up").addClass("topbar-down");
      if (toc.length > 0) {
        toc.addClass("topbar-down");
      }
      if (access.length > 0) {
        access.addClass("topbar-down");
=======
    if (st > lastScrollTop ) { // Scroll Down
      if (st > topbarHeight) {
        $topbarWrapper.removeClass(CLASS_TOPBAR_DOWN).addClass(CLASS_TOPBAR_UP);
        $panel.removeClass(CLASS_TOPBAR_DOWN);

        if ($searchInput.is(":focus")) {
          $searchInput.blur(); /* remove focus */
        }
      }
    } else  {// Scroll up
      // did not reach the bottom of the document, i.e., still have space to scroll up
      if (st + $(window).height() < $(document).height()) {
        let tocScrollingUp = $topbarWrapper.attr(ATTR_TOC_SCROLLING_UP);
        if (typeof tocScrollingUp !== "undefined") {
          if (tocScrollingUp === "false") {
            $topbarWrapper.removeAttr(ATTR_TOC_SCROLLING_UP);
          }

        } else {
          $topbarWrapper.removeClass(CLASS_TOPBAR_UP).addClass(CLASS_TOPBAR_DOWN);
          $panel.addClass(CLASS_TOPBAR_DOWN);
        }
>>>>>>> ebb3dc940c22d864dc41a16f1d84c1a0c0a003ba
      }
    }

    lastScrollTop = st;
  }

  $(window).scroll(function(event) {
<<<<<<< HEAD
    if ($("#topbar-title").is(":hidden")) { /* Not in small screens */
=======
    if ($("#topbar-title").is(":hidden")) {
>>>>>>> ebb3dc940c22d864dc41a16f1d84c1a0c0a003ba
      didScroll = true;
    }
  });

  setInterval(function() {
    if (didScroll) {
      hasScrolled();
      didScroll = false;
    }
  }, 250);

});
