/**
 *  Add language indicator to code snippets
 */

$(function() {
  const prefix = "language-";
  const regex = new RegExp(`^${prefix}([a-z])+$`);

  $(`div[class^=${prefix}`).each(function() {
    let classes = $(this).attr("class").split(" ");

    classes.forEach((_class) => {
      if (regex.test(_class)) {
        let lang = _class.substring(prefix.length);
        $(this).attr("lang", `${lang}`);
      }
    });

  });
});
