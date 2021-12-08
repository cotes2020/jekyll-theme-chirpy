/*
 * Hide Header on scroll down
 */

$(function() {
  const $topbarWrapper = $("#topbar-wrapper");
  const $panel = $("#panel-wrapper");
  const $searchInput = $("#search-input");

  const CLASS_TOPBAR_UP = "topbar-up";
  const CLASS_TOPBAR_DOWN = "topbar-down";
  const ATTR_TOC_SCROLLING_UP = "toc-scrolling-up"; // topbar locked

  let didScroll;
  let lastScrollTop = 0;

  const delta = $topbarWrapper.outerHeight();
  const topbarHeight = $topbarWrapper.outerHeight();

  function hasScrolled() {
    let st = $(this).scrollTop();

    /* Make sure they scroll more than delta */
    if (Math.abs(lastScrollTop - st) <= delta) {
      return;
    }

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
      }
    }

    lastScrollTop = st;
  }

  $(window).scroll(function(event) {
    if ($("#topbar-title").is(":hidden")) {
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
