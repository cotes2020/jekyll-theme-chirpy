/*
 * Copy current page url to clipboard.
 */

function copyLink(url, msg) {
  if (!url || 0 === url.length) {
    url = window.location.href;
  }

  const $temp = $("<input>");
  $("body").append($temp);
  $temp.val(url).select();
  document.execCommand("copy");
  $temp.remove();

  let feedback = "Link copied successfully!";
  if (msg && msg.length > 0) {
    feedback = msg;
  }

  alert(feedback);
}
