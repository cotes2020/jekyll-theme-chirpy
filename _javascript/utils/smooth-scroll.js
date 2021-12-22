/*
  Safari doesn't support CSS `scroll-behavior: smooth`,
  so here is a compatible solution for all browser to smooth scrolling

  See: <https://css-tricks.com/snippets/jquery/smooth-scrolling/>

  Warning: It must be called after all `<a>` tags (e.g., the dynamic TOC) are ready.
*/

$(function() {
<<<<<<< HEAD
=======
  const $topbarWrapper = $("#topbar-wrapper");
  const topbarHeight = $topbarWrapper.outerHeight();
  const $topbarTitle = $("#topbar-title");

  const ATTR_TOC_SCROLLING = "toc-scrolling-up";
  const SCROLL_MARK = "scroll-focus";
  const REM = 16; // in pixels
  let tocScrollUpCount = 0;

>>>>>>> ebb3dc940c22d864dc41a16f1d84c1a0c0a003ba
  $("a[href*='#']")
    .not("[href='#']")
    .not("[href='#0']")
    .click(function(event) {
<<<<<<< HEAD

      if (this.pathname.replace(/^\//, "") === location.pathname.replace(/^\//, "")) {
        if (location.hostname === this.hostname) {

          const REM = 16; /* 16px */

          const hash = decodeURI(this.hash);
          let isFnRef = RegExp(/^#fnref:/).test(hash);
          let isFn = RegExp(/^#fn:/).test(hash);
          let selector = hash.includes(":") ? hash.replace(/\:/, "\\:") : hash;
          let target = $(selector);

          if (target.length) {
=======
      if (this.pathname.replace(/^\//, "") === location.pathname.replace(/^\//, "")) {
        if (location.hostname === this.hostname) {
          const hash = decodeURI(this.hash);
          let toFootnoteRef = RegExp(/^#fnref:/).test(hash);
          let toFootnote = toFootnoteRef? false : RegExp(/^#fn:/).test(hash);
          let selector = hash.includes(":") ? hash.replace(/\:/g, "\\:") : hash;
          let $target = $(selector);

          let parent = $(this).parent().prop("tagName");
          let isAnchor = RegExp(/^H\d/).test(parent);
          let isMobileViews = !$topbarTitle.is(":hidden");

          if (typeof $target !== "undefined") {
>>>>>>> ebb3dc940c22d864dc41a16f1d84c1a0c0a003ba
            event.preventDefault();

            if (history.pushState) { /* add hash to URL */
              history.pushState(null, null, hash);
            }

<<<<<<< HEAD
            let curOffset = $(this).offset().top;
            let destOffset = target.offset().top;
            const scrollUp = (destOffset < curOffset);
            const topbarHeight = $("#topbar-wrapper").outerHeight();

            if (scrollUp && isFnRef) {
              /* Avoid the top-bar covering `fnref` when scrolling up
                because `fnref` has no `%anchor`(see: module.scss) style. */
              destOffset -= (topbarHeight + REM / 2);
            }

            $("html,body").animate({
              scrollTop: destOffset
            }, 800, () => {

              const $target = $(target);
              $target.focus();

              const SCROLL_MARK = "scroll-focus";

=======
            let curOffset = isAnchor? $(this).offset().top : $(window).scrollTop();
            let destOffset = $target.offset().top -= REM / 2;

            if (destOffset < curOffset) { // scroll up
              if (!isAnchor && !toFootnote) { // trigger by ToC item
                if (!isMobileViews) { // on desktop/tablet screens
                  $topbarWrapper.removeClass("topbar-down").addClass("topbar-up");
                  // Send message to `${JS_ROOT}/commons/topbar-switch.js`
                  $topbarWrapper.attr(ATTR_TOC_SCROLLING, true);
                  tocScrollUpCount += 1;
                }
              }

              if ((isAnchor || toFootnoteRef) && isMobileViews) {
                destOffset -= topbarHeight;
                console.log(`[smooth] mobile -= topbar height`);
              }
            }

            $("html").animate({
              scrollTop: destOffset
            }, 500, () => {
              $target.focus();

>>>>>>> ebb3dc940c22d864dc41a16f1d84c1a0c0a003ba
              /* clean up old scroll mark */
              if ($(`[${SCROLL_MARK}=true]`).length) {
                $(`[${SCROLL_MARK}=true]`).attr(SCROLL_MARK, false);
              }

              /* Clean :target links */
              if ($(":target").length) { /* element that visited by the URL with hash */
                $(":target").attr(SCROLL_MARK, false);
              }

              /* set scroll mark to footnotes */
<<<<<<< HEAD
              if (isFn || isFnRef) {
=======
              if (toFootnote || toFootnoteRef) {
>>>>>>> ebb3dc940c22d864dc41a16f1d84c1a0c0a003ba
                $target.attr(SCROLL_MARK, true);
              }

              if ($target.is(":focus")) { /* Checking if the target was focused */
                return false;
              } else {
                $target.attr("tabindex", "-1"); /* Adding tabindex for elements not focusable */
                $target.focus(); /* Set focus again */
              }
<<<<<<< HEAD
=======

              if (typeof $topbarWrapper.attr(ATTR_TOC_SCROLLING) !== "undefined") {
                tocScrollUpCount -= 1;

                if (tocScrollUpCount <= 0) {
                  $topbarWrapper.attr(ATTR_TOC_SCROLLING, "false");
                }
              }
>>>>>>> ebb3dc940c22d864dc41a16f1d84c1a0c0a003ba
            });
          }
        }
      }

    }); /* click() */
});
