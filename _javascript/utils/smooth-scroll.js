/**
 Safari doesn't support CSS `scroll-behavior: smooth`,
 so here is a compatible solution for all browser to smooth scrolling

 See: <https://css-tricks.com/snippets/jquery/smooth-scrolling/>

 Warning: It must be called after all `<a>` tags (e.g., the dynamic TOC) are ready.
 */

$(function () {
    const $topbarTitle = $("#topbar-title");
    const REM = 16; // in pixels
    const ATTR_SCROLL_FOCUS = "scroll-focus";

    $("a[href*='#']")
        .not("[href='#']")
        .not("[href='#0']")
        .on('click', function (event) {
            if (this.pathname.replace(/^\//, "") !==
                location.pathname.replace(/^\//, "")) {
                return;
            }

            if (location.hostname !== this.hostname) {
                return;
            }

            const hash = decodeURI(this.hash);
            let toFootnoteRef = RegExp(/^#fnref:/).test(hash);
            let toFootnote = toFootnoteRef ? false : RegExp(/^#fn:/).test(hash);
            let selector = '#' + $.escapeSelector(hash.substring(1));
            let $target = $(selector);

            let isMobileViews = $topbarTitle.is(":visible");
            let isPortrait = $(window).width() < $(window).height();

            if (typeof $target === "undefined") {
                return;
            }

            event.preventDefault();

            if (history.pushState) { /* add hash to URL */
                history.pushState(null, null, hash);
            }

            let curOffset = $(window).scrollTop();
            let destOffset = $target.offset().top -= REM / 2;

            if (destOffset < curOffset) { // scroll up
                ScrollHelper.hideTopbar();
                ScrollHelper.addScrollUpTask();

                if (isMobileViews && isPortrait) {
                    destOffset -= ScrollHelper.getTopbarHeight();
                }

            } else { // scroll down
                if (isMobileViews && isPortrait) {
                    destOffset -= ScrollHelper.getTopbarHeight();
                }
            }

            $("html").animate({
                scrollTop: destOffset
            }, 500, () => {
                $target.trigger("focus");

                /* clean up old scroll mark */
                const $scroll_focus = $(`[${ATTR_SCROLL_FOCUS}=true]`);
                if ($scroll_focus.length) {
                    $scroll_focus.attr(ATTR_SCROLL_FOCUS, "false");
                }

                /* Clean :target links */
                const $target_links = $(":target");
                if ($target_links.length) { /* element that visited by the URL with hash */
                    $target_links.attr(ATTR_SCROLL_FOCUS, "false");
                }

                /* set scroll mark to footnotes */
                if (toFootnote || toFootnoteRef) {
                    $target.attr(ATTR_SCROLL_FOCUS, "true");
                }

                if ($target.is(":focus")) { /* Checking if the target was focused */
                    return false;
                } else {
                    $target.attr("tabindex", "-1"); /* Adding tabindex for elements not focusable */
                    $target.trigger("focus"); /* Set focus again */
                }

                if (ScrollHelper.hasScrollUpTask()) {
                    ScrollHelper.popScrollUpTask();
                }
            });
        }); /* click() */
});
