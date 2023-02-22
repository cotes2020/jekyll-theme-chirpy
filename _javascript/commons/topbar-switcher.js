/**
 * Hide Header on scroll down
 */

$(function () {
    const $searchInput = $("#search-input");
    const delta = ScrollHelper.getTopbarHeight();

    let didScroll;
    let lastScrollTop = 0;

    function hasScrolled() {
        let st = $(this).scrollTop();

        /* Make sure they scroll more than delta */
        if (Math.abs(lastScrollTop - st) <= delta) {
            return;
        }

        if (st > lastScrollTop) { // Scroll Down
            ScrollHelper.hideTopbar();

            if ($searchInput.is(":focus")) {
                $searchInput.trigger('blur'); /* remove focus */
            }

        } else { // Scroll up
            // has not yet scrolled to the bottom of the screen, that is, there is still space for scrolling
            if (st + $(window).height() < $(document).height()) {

                if (ScrollHelper.hasScrollUpTask()) {
                    return;
                }

                if (ScrollHelper.topbarLocked()) { // avoid redundant scroll up event from smooth scrolling
                    ScrollHelper.unlockTopbar();
                } else {
                    if (ScrollHelper.orientationLocked()) { // avoid device auto scroll up on orientation change
                        ScrollHelper.unLockOrientation();
                    } else {
                        ScrollHelper.showTopbar();
                    }
                }
            }
        }

        lastScrollTop = st;

    } // hasScrolled()

    function handleLandscape() {
        if ($(window).scrollTop() === 0) {
            return;
        }
        ScrollHelper.lockOrientation();
        ScrollHelper.hideTopbar();
    }

    if (screen.orientation) {
        screen.orientation.onchange = () => {
            const type = screen.orientation.type;
            if (type === "landscape-primary" || type === "landscape-secondary") {
                handleLandscape();
            }
        };

    } else {
        // for the browsers that not support `window.screen.orientation` API
        $(window).on("orientationchange", () => {
            if ($(window).width() < $(window).height()) { // before rotating, it is still in portrait mode.
                handleLandscape();
            }
        });
    }

    $(window).on('scroll',() => {
        if (didScroll) {
            return;
        }
        didScroll = true;
    });

    setInterval(() => {
        if (didScroll) {
            hasScrolled();
            didScroll = false;
        }
    }, 250);
});
