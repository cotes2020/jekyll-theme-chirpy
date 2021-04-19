/*
Reference: https://bootsnipp.com/snippets/featured/link-to-top-page
*/
$(function() {
  $(window).scroll(() => {
    if ($(this).scrollTop() > 50 &&
      $("#sidebar-trigger").css("display") === "none") {
      $("#back-to-top").fadeIn();
    } else {
      $("#back-to-top").fadeOut();
    }
  });

  $("#back-to-top").click(() => {
    $("body,html").animate({
      scrollTop: 0
    }, 800);
    return false;
  });
});
