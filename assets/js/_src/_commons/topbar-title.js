/*
* Topbar title auto change while scrolling in mobile screens.
* © 2018-2019 Cotes Chung
* MIT License
*/
$(function(){

  var DEFAULT = $("#topbar-title").text().trim();
  var title = ($("div.post>h1").length > 0) ?
          $("div.post>h1").text().trim() : $("h1").text().trim();

  if ($("#page-category").length || $("#page-tag").length) {
    /* The title in Category or Tag page will be '<title> <count_of_posts>' */
    if (/\s/.test(title)) {
      title = title.replace(/[0-9]/g, '').trim();
    }
  }

  // Replace topbar title while scroll screens.
  $(window).scroll(function () {
    if ($("#post-list").length // in Home page
      || $("div.post>h1").is(":hidden") // is tab pages
      || $("#topbar-title").is(":hidden") // not mobile screens
      || $("#sidebar.sidebar-expand").length) { // when the sidebar trigger is clicked
      return false;
    }

    if ($(this).scrollTop() >= 95) {
      if ($("#topbar-title").text() != title) {
        $("#topbar-title").text(title);
      }
    } else {
      if ($("#topbar-title").text() != DEFAULT) {
        $("#topbar-title").text(DEFAULT);
      }
    }
  })

  // Click title remove hover effect.
  $('#topbar-title').click(function() {
    $('body,html').animate({scrollTop: 0}, 800);
  });

});