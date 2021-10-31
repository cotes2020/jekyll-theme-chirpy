/*
  Safari doesn't support CSS `scroll-behavior: smooth`,
  so here is a compatible solution for all browser to smooth scrolling

  See: <https://css-tricks.com/snippets/jquery/smooth-scrolling/>

  Warning: It must be called after all `<a>` tags (e.g., the dynamic TOC) are ready.
*/

$(function() {
  $("a[href*='#']")
    .not("[href='#']")
    .not("[href='#0']")
    .click(function(event) {

      if (this.pathname.replace(/^\//, "") === location.pathname.replace(/^\//, "")) {
        if (location.hostname === this.hostname) {

          const REM = 16; /* 16px */

          const hash = decodeURI(this.hash);
          let isFnRef = RegExp(/^#fnref:/).test(hash);
          let isFn = isFnRef? false : RegExp(/^#fn:/).test(hash);
          let selector = hash.includes(":") ? hash.replace(/\:/g, "\\:") : hash;
          let target = $(selector);

          if (target.length) {
            event.preventDefault();

            if (history.pushState) { /* add hash to URL */
              history.pushState(null, null, hash);
            }

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

              /* clean up old scroll mark */
              if ($(`[${SCROLL_MARK}=true]`).length) {
                $(`[${SCROLL_MARK}=true]`).attr(SCROLL_MARK, false);
              }

              /* Clean :target links */
              if ($(":target").length) { /* element that visited by the URL with hash */
                $(":target").attr(SCROLL_MARK, false);
              }

              /* set scroll mark to footnotes */
              if (isFn || isFnRef) {
                $target.attr(SCROLL_MARK, true);
              }

              if ($target.is(":focus")) { /* Checking if the target was focused */
                return false;
              } else {
                $target.attr("tabindex", "-1"); /* Adding tabindex for elements not focusable */
                $target.focus(); /* Set focus again */
              }
            });
          }
        }
      }

    }); /* click() */
});
