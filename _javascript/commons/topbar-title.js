/*
 * Top bar title auto change while scrolling in mobile screens.
*/

$(function() {

  const topbarTitle = $("#topbar-title");
  const postTitle = $("div.post>h1");

  const DEFAULT = topbarTitle.text().trim();

  let title = (postTitle.length > 0) ?
    postTitle.text().trim() : $("h1").text().trim();

  if ($("#page-category").length || $("#page-tag").length) {
    /* The title in Category or Tag page will be "<title> <count_of_posts>" */
    if (/\s/.test(title)) {
      title = title.replace(/[0-9]/g, "").trim();
    }
  }

  /* Replace topbar title while scroll screens. */
  $(window).scroll(function () {
    if ($("#post-list").length /* in Home page */
      || postTitle.is(":hidden") /* is tab pages */
      || topbarTitle.is(":hidden") /* not mobile screens */
      || $("#sidebar.sidebar-expand").length) { /* when the sidebar trigger is clicked */
      return false;
    }

    if ($(this).scrollTop() >= 95) {
      if (topbarTitle.text() !== title) {
        topbarTitle.text(title);
      }
    } else {
      if (topbarTitle.text() !== DEFAULT) {
        topbarTitle.text(DEFAULT);
      }
    }
  });

  /* Click title remove hover effect. */
  topbarTitle.click(function() {
    $("body,html").animate({scrollTop: 0}, 800);
  });

});
