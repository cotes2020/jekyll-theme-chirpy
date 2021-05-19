/*
 * Invoke iOS/Android share sheet
 */

function shareLink(title, text, url) {
    if (!url || 0 === url.length) {
        url = window.location.href;
    }

    if (navigator.share) {
        navigator.share({
            title: title,
            text: text,
            url: url,
          })
          .then(() => console.log('Successful share'))
          .catch((error) => console.log('Error sharing', error));
    } else {
        const $temp = $("<input>");
        $("body").append($temp);
        $temp.val(url).select();
        document.execCommand("copy");
        $temp.remove();
        alert("Your browser doesn't support direct share, Page link coped to clipboard instead.");
        return;
    }
}