/*
 * Calculate the Timeago
 */

$(function() {
<<<<<<< HEAD

  const timeagoElem = $(".timeago");

  let toRefresh = timeagoElem.length;

  let intervalId = void 0;

  function timeago(iso, preposition) {
    let now = new Date();
    let past = new Date(iso);
    let prep = (typeof preposition !== "undefined" ? `${preposition} ` : "");

    if (past.getFullYear() !== now.getFullYear()) {
      toRefresh -= 1;
      return prep + past.toLocaleString("en-US", {
        year: "numeric",
        month: "short",
        day: "numeric"
      });
    }

    if (past.getMonth() !== now.getMonth()) {
      toRefresh -= 1;
      return prep + past.toLocaleString("en-US", {
        month: "short",
        day: "numeric"
      });
=======
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
>>>>>>> ebb3dc940c22d864dc41a16f1d84c1a0c0a003ba
    }

    let seconds = Math.floor((now - past) / 1000);

    let day = Math.floor(seconds / 86400);
    if (day >= 1) {
<<<<<<< HEAD
      toRefresh -= 1;
      return day + " day" + (day > 1 ? "s" : "") + " ago";
=======
      return ` ${day} ${dPrompt}`;
>>>>>>> ebb3dc940c22d864dc41a16f1d84c1a0c0a003ba
    }

    let hour = Math.floor(seconds / 3600);
    if (hour >= 1) {
<<<<<<< HEAD
      return hour + " hour" + (hour > 1 ? "s" : "") + " ago";
=======
      return ` ${hour} ${hrPrompt}`;
>>>>>>> ebb3dc940c22d864dc41a16f1d84c1a0c0a003ba
    }

    let minute = Math.floor(seconds / 60);
    if (minute >= 1) {
<<<<<<< HEAD
      return minute + " minute" + (minute > 1 ? "s" : "") + " ago";
    }

    return "just now";
=======
      return ` ${minute} ${minPrompt}`;
    }

    return justnowPrompt;
>>>>>>> ebb3dc940c22d864dc41a16f1d84c1a0c0a003ba
  }

  function updateTimeago() {
    $(".timeago").each(function() {
<<<<<<< HEAD
      if ($(this).children("i").length > 0) {
        let node = $(this).children("i");
        let date = node.text(); /* ISO Date: "YYYY-MM-DDTHH:MM:SSZ" */
        $(this).text(timeago(date, $(this).attr("prep")));
        $(this).append(node);
      }
    });

    if (toRefresh === 0 && typeof intervalId !== "undefined") {
      clearInterval(intervalId); /* stop interval */
    }
    return toRefresh;
  }

  if (toRefresh === 0) {
=======
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
>>>>>>> ebb3dc940c22d864dc41a16f1d84c1a0c0a003ba
    return;
  }

  if (updateTimeago() > 0) { /* run immediately */
    intervalId = setInterval(updateTimeago, 60000); /* run every minute */
  }

});
