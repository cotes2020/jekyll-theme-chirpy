/*
  Safari doesn't support CSS `scroll-behavior: smooth`,
  so here is a compatible solution for all browser to smooth scrolling

  See: <https://css-tricks.com/snippets/jquery/smooth-scrolling/>

  Warning: It must be called after all `<a>` tags (e.g., the dynamic TOC) are ready.
*/

$(function() {
  const $topbarTitle = $("#topbar-title");
  const topbarHeight = $("#topbar-wrapper").outerHeight();
  const SCROLL_MARK = "scroll-focus";

  $("a[href*='#']")
    .not("[href='#']")
    .not("[href='#0']")
    .click(function(event) {

      if (this.pathname.replace(/^\//, "") === location.pathname.replace(/^\//, "")) {
        if (location.hostname === this.hostname) {
          const hash = decodeURI(this.hash);
          let toFootnoteRef = RegExp(/^#fnref:/).test(hash);
          let toFootnote = toFootnoteRef? false : RegExp(/^#fn:/).test(hash);
          let selector = hash.includes(":") ? hash.replace(/\:/g, "\\:") : hash;
          let $target = $(selector);

          let parent = $(this).parent().prop("tagName");
          let isAnchor = RegExp(/^H\d/).test(parent);

          if (typeof $target !== "undefined") {
            event.preventDefault();

            if (history.pushState) { /* add hash to URL */
              history.pushState(null, null, hash);
            }

            let curOffset = isAnchor? $(this).offset().top : $(window).scrollTop();
            let destOffset = $target.offset().top;

            if (destOffset < curOffset) { // scroll up
              if (!isAnchor && !toFootnote && $topbarTitle.is(":hidden")) { // the ToC item
                destOffset -= topbarHeight;
              }
            }

            $("html,body").animate({
              scrollTop: destOffset

            }, 800, () => {
              $target.focus();

              /* clean up old scroll mark */
              if ($(`[${SCROLL_MARK}=true]`).length) {
                $(`[${SCROLL_MARK}=true]`).attr(SCROLL_MARK, false);
              }

              /* Clean :target links */
              if ($(":target").length) { /* element that visited by the URL with hash */
                $(":target").attr(SCROLL_MARK, false);
              }

              /* set scroll mark to footnotes */
              if (toFootnote || toFootnoteRef) {
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
