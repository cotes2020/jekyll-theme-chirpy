// Code snippet inspired by https://github.com/douglasrodrigues5/ghost-blog-infinite-scroll
$(function ($) {
    var currentPage = 1;
    var pathname = window.location.pathname;
    var $document = $(document);
    var $result = $('.post-feed');
    var buffer = 100;

    var ticking = false;
    var isLoading = false;

    var lastScrollY = window.scrollY;
    var lastWindowHeight = window.innerHeight;
    var lastDocumentHeight = $document.height();

    // remove hash params from pathname
    pathname = pathname.replace(/#(.*)$/g, '').replace('/\//g', '/');

    function onScroll() {
        lastScrollY = window.scrollY;
        requestTick();
    }

    function onResize() {
        lastWindowHeight = window.innerHeight;
        lastDocumentHeight = $document.height();
        requestTick();
    }

    function requestTick() {
        if (!ticking) {
            requestAnimationFrame(infiniteScroll)
        }
        ticking = true;
    }

    function infiniteScroll () {
        // return if already loading
        if (isLoading) {
            return;
        }

        // return if not scroll to the bottom
        if (lastScrollY + lastWindowHeight <= lastDocumentHeight - buffer) {
            ticking = false;
            return;
        }

        // return if currentPage is the last page already
        if (currentPage === maxPages) {
            return;
        }

        isLoading = true;

        // next page
        currentPage++;

        // Load more
        var nextPage = pathname + 'page' + currentPage + '/';

        $.get(nextPage, function (content) {
            $result.append($(content).find('.post').hide().fadeIn(100));

        }).fail(function (xhr) {
            // 404 indicates we've run out of pages
            if (xhr.status === 404) {
                window.removeEventListener('scroll', onScroll, {passive: true});
                window.removeEventListener('resize', onResize);
            }

        }).always(function () {
            lastDocumentHeight = $document.height();
            isLoading = false;
            ticking = false;
        });
    }

    window.addEventListener('scroll', onScroll, {passive: true});
    window.addEventListener('resize', onResize);

    infiniteScroll();
});
