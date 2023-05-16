/**
 * Reference: https://bootsnipp.com/snippets/featured/link-to-top-page
 */

export function back2top() {
  $(window).on('scroll', () => {
    if ($(window).scrollTop() > 50) {
      $('#back-to-top').fadeIn();
    } else {
      $('#back-to-top').fadeOut();
    }
  });

  $('#back-to-top').on('click', () => {
    window.scrollTo(0, 0);
  });
}
