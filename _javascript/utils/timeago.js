/*
 * Calculate the Timeago
 */

$(function() {
  const timeagoElem = $(".timeago");
  let tasks = timeagoElem.length;
  let intervalId = void 0;

  const dPrompt = $("meta[name=day-prompt]").attr("content");
  const hrPrompt = $("meta[name=hour-prompt]").attr("content");
  const minPrompt = $("meta[name=minute-prompt]").attr("content");
  const justnowPrompt = $("meta[name=justnow-prompt]").attr("content");

  function timeago(date, initDate) {
    let now = new Date();
    let past = new Date(date);

    if (past.getFullYear() !== now.getFullYear()
        || past.getMonth() !== now.getMonth()) {
      return initDate;
    }

    let seconds = Math.floor((now - past) / 1000);

    let day = Math.floor(seconds / 86400);
    if (day >= 1) {
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
      if ($(this)[0].hasAttribute("date") === false) {
        tasks -= 1;
        return;
      }

      let date = $(this).attr("date");
      let initDate = $(this).text();
      let relativeDate = timeago(date, initDate);

      if (relativeDate === initDate) {
        $(this).removeAttr("date");
      } else {
        $(this).text(relativeDate);
      }

    });

    if (tasks === 0 && typeof intervalId !== "undefined") {
      clearInterval(intervalId); /* stop interval */
    }
    return tasks;
  }

  if (tasks === 0) {
    return;
  }

  if (updateTimeago() > 0) { /* run immediately */
    intervalId = setInterval(updateTimeago, 60000); /* run every minute */
  }

});
