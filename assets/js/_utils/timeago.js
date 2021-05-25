/*
 * Caculate the Timeago
 * v2.0
 * https://github.com/cotes2020/jekyll-theme-chirpy
 * © 2019 Cotes Chung
 * MIT Licensed
 */

$(function() {

  var toRefresh = $(".timeago").length;

  var intervalId = void 0;

  function timeago(iso, isLastmod) {
    let now = new Date();
    let past = new Date(iso);

    if (past.getFullYear() !== now.getFullYear()) {
      toRefresh -= 1;
      return past.toLocaleString("zh-CN", {
        year: "numeric",
        month: "short",
        day: "numeric"
      });
    }

    if (past.getMonth() !== now.getMonth()) {
      toRefresh -= 1;
      return past.toLocaleString("zh-CN", {
        month: "short",
        day: "numeric"
      });
    }

    let seconds = Math.floor((now - past) / 1000);

    let day = Math.floor(seconds / 86400);
    if (day >= 1) {
      toRefresh -= 1;
      return day + " 天前";
    }

    let hour = Math.floor(seconds / 3600);
    if (hour >= 1) {
      return hour + " 小时前";
    }

    let minute = Math.floor(seconds / 60);
    if (minute >= 1) {
      return minute + " 分钟前";
    }

    return "刚刚发布";
  }

  function updateTimeago() {
    $(".timeago").each(function() {
      if ($(this).children("i").length > 0) {
        var basic = $(this).text();
        var isLastmod = $(this).hasClass("lastmod");
        var node = $(this).children("i");
        var date = node.text(); /* ISO Date: "YYYY-MM-DDTHH:MM:SSZ" */
        $(this).text(timeago(date, isLastmod));
        $(this).append(node);
      }
    });

    if (toRefresh === 0 && typeof intervalId !== "undefined") {
      clearInterval(intervalId); /* stop interval */
    }
    return toRefresh;
  }

  if (toRefresh === 0) {
    return;
  }

  if (updateTimeago() > 0) { /* run immediately */
    intervalId = setInterval(updateTimeago, 60000); /* run every minute */
  }

});
