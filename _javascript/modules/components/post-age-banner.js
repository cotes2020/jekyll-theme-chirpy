export function postAgeBanner() {

  // Assign the target node containing the HTML message to a variable
  let postBanner = document.getElementById('post-age-banner');

  // Read the contents of the 'div' containing the posted date
  let postDate = Number(document.getElementById('posted-date').innerHTML);

  // Read the contents of the 'div' containing the post's last update date
  let updatedDate = Number(document.getElementById('updated-date').innerHTML);

  // Read the contents of the 'div' containing the post's last update date
  let ageMonths = Number(document.getElementById('age-months').innerHTML);


  // Check if the updated date is newer than the posted date
  // if it is then use the updated date, else use the postDate
  let postCheckDate = updatedDate > postDate ? updatedDate : postDate;

  // Get the current age of the post by subtracting the post's timestamp from the current UNIX timestamp when the page is loaded, i.e. right now
  let postAge = Math.floor(new Date().getTime() / 1000.0) - postCheckDate;

  // Calculate the age in months by dividing the post's age by the number of seconds in a month, and rounding it down to an integer
  let postAgeMonths = Math.floor(postAge / 2629746);


  // If the post age (in months) is greater than 6 (months), then remove the hidden CSS class from the target node, and insert the HTML message
  if (postAgeMonths > ageMonths) {
    postBanner.classList.remove('d-none');
    postBanner.textContent = `This post is ${postAgeMonths} months old. Some information may be outdated or inaccurate.`;
  }
}
