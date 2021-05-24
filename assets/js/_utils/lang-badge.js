/**
 *  Add language indicator to code snippets
 */

$(function() {
  const prefix = "language-";
  const regex = new RegExp(`^${prefix}([a-z])+$`);

  $(`div[class^=${prefix}`).each(function() {
    let clzsses = $(this).attr("class").split(" ");

    clzsses.forEach((clzss) => {
      if (regex.test(clzss)) {
        let lang = clzss.substring(prefix.length);
        $(this).attr("lang", `${lang}`);
      }
    });

  });
});
