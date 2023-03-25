/**
 * Reference: https://bootsnipp.com/snippets/featured/link-to-top-page
 */

export function back2top() {
  $(window).on('scroll', () => {
    if (
      $(window).scrollTop() > 50 &&
      $('#sidebar-trigger').css('display') === 'none'
    ) {
      $('#back-to-top').fadeIn();
    } else {
      $('#back-to-top').fadeOut();
    }
  });

  $('#back-to-top').on('click', () => {
    $('body,html').animate(
      {
        scrollTop: 0
      },
      800
    );
    return false;
  });
}
