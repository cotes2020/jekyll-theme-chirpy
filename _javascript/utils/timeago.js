/*
 * Calculate the Timeago
 */

$(function() {

  const timeagoElem = $(".timeago");

  let toRefresh = timeagoElem.length;

  let intervalId = void 0;

  const locale = $("meta[name=layout-lang]").attr("content");
  const dPrompt = $("meta[name=day-prompt]").attr("content");
  const hrPrompt = $("meta[name=hour-prompt]").attr("content");
  const minPrompt = $("meta[name=minute-prompt]").attr("content");
  const justnowPrompt = $("meta[name=justnow-prompt]").attr("content");

  function timeago(isoDate, dateStr) {
    let now = new Date();
    let past = new Date(isoDate);

    if (past.getFullYear() !== now.getFullYear()
        || past.getMonth() !== now.getMonth()) {
      return dateStr;
    }

    let seconds = Math.floor((now - past) / 1000);

    let day = Math.floor(seconds / 86400);
    if (day >= 1) {
      toRefresh -= 1;
      return ` ${day} ${dPrompt}`;
    }

    let hour = Math.floor(seconds / 3600);
    if (hour >= 1) {
      return ` ${hour} ${hrPrompt}`;
    }

    let minute = Math.floor(seconds / 60);
    if (minute >= 1) {
      return ` ${minute} ${minPrompt}`;
    }

    return justnowPrompt;
  }

  function updateTimeago() {
    $(".timeago").each(function() {
      if ($(this).children("i").length > 0) {
        let dateStr = $(this).clone().children().remove().end().text();
        let node = $(this).children("i");
        let iosDate = node.text(); /* ISO Date: "YYYY-MM-DDTHH:MM:SSZ" */
        $(this).text(timeago(iosDate, dateStr));
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
