/**
 * Top bar title auto change while scrolling up/down in mobile screens.
 */
const titleSelector = 'div.post>h1:first-of-type';
const $pageTitle = $(titleSelector);
const $topbarTitle = $('#topbar-title');
const defaultTitleText = $topbarTitle.text().trim();

export function convertTitle() {
  if (
    $pageTitle.length === 0 /* on Home page */ ||
    $pageTitle.hasClass('dynamic-title') ||
    $topbarTitle.is(':hidden')
  ) {
    /* not in mobile views */
    return;
  }

  let pageTitleText = $pageTitle.text().trim();
  let hasScrolled = false;
  let lastScrollTop = 0;

  if ($('#page-category').length || $('#page-tag').length) {
    /* The title in Category or Tag page will be "<title> <count_of_posts>" */
    if (/\s/.test(pageTitleText)) {
      pageTitleText = pageTitleText.replace(/[0-9]/g, '').trim();
    }
  }

  // When the page is scrolled down and then refreshed, the topbar title needs to be initialized
  if ($pageTitle.offset().top < $(window).scrollTop()) {
    $topbarTitle.text(pageTitleText);
  }

  let options = {
    rootMargin: '-48px 0px 0px 0px', // 48px equals to the topbar height (3rem)
    threshold: [0, 1]
  };

  let observer = new IntersectionObserver((entries) => {
    if (!hasScrolled) {
      hasScrolled = true;
      return;
    }

    let curScrollTop = $(window).scrollTop();
    let isScrollDown = lastScrollTop < curScrollTop;
    lastScrollTop = curScrollTop;
    let heading = entries[0];

    if (isScrollDown) {
      if (heading.intersectionRatio === 0) {
        $topbarTitle.text(pageTitleText);
      }
    } else {
      if (heading.intersectionRatio === 1) {
        $topbarTitle.text(defaultTitleText);
      }
    }
  }, options);

  observer.observe(document.querySelector(titleSelector));

  /* Click title will scroll to top */
  $topbarTitle.on('click', function () {
    $('body,html').animate({ scrollTop: 0 }, 800);
  });
}
