/*
 * Caculate the Timeago
 * v2.0
 * https://github.com/cotes2020/jekyll-theme-chirpy
 * © 2019 Cotes Chung
 * MIT Licensed
 */

$(function() {

  function timeago(iso, isLastmod) {
    let now = new Date();
    let past = new Date(iso);

    if (past.getFullYear() != now.getFullYear()) {
      toRefresh -= 1;
      return past.toLocaleString("en-US", {
         year: 'numeric',
         month: 'short',
         day: 'numeric'
      });
    }

    if (past.getMonth() != now.getMonth()) {
      toRefresh -= 1;
      return past.toLocaleString("en-US", {
         month: 'short',
         day: 'numeric'
      });
    }

    let seconds = Math.floor((now - past) / 1000);

    let day = Math.floor(seconds / 86400);
    if (day >= 1) {
      toRefresh -= 1;
      return day + " day" + (day > 1 ? "s" : "") + " ago";
    }

    let hour = Math.floor(seconds / 3600);
    if (hour >= 1) {
      return hour + " hour" + (hour > 1 ? "s" : "") + " ago";
    }

    let minute = Math.floor(seconds / 60);
    if (minute >= 1) {
      return minute + " minute" + (minute > 1 ? "s" : "") + " ago";
    }

    return (isLastmod? "just" : "Just") + " now";
  }


  function updateTimeago() {
    $(".timeago").each(function() {
      if ($(this).children("i").length > 0) {
        var basic = $(this).text();
        var isLastmod = $(this).hasClass('lastmod');
        var node = $(this).children("i");
        var date = node.text();   /* ISO Date: 'YYYY-MM-DDTHH:MM:SSZ' */
        $(this).text(timeago(date, isLastmod));
        $(this).append(node);
      }
    });

    if (toRefresh == 0 && intervalId != undefined) {
      clearInterval(intervalId);  /* stop interval */
    }
    return toRefresh;
  }


  var toRefresh = $(".timeago").length;

  if (toRefresh == 0) {
    return;
  }

  if (updateTimeago() > 0) { /* run immediately */
    var intervalId = setInterval(updateTimeago, 60000); /* run every minute */
  }

});