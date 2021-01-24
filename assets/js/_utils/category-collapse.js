/*
 * Tab 'Categories' expand/close effect.
 */

$(function() {
  const childPrefix = "l_";
  const parentPrefix = "h_";
  const collapse = $(".collapse");

  /* close up top-category */
  collapse.on("hide.bs.collapse", function () { /* Bootstrap collapse events. */
    const parentId = parentPrefix + $(this).attr("id").substring(childPrefix.length);
    if (parentId) {
      $(`#${parentId} .far.fa-folder-open`).attr("class", "far fa-folder fa-fw");
      $(`#${parentId} i.fas`).addClass("rotate");
      $(`#${parentId}`).removeClass("hide-border-bottom");
    }
  });

  /* expand the top category */
  collapse.on("show.bs.collapse", function() {
    const parentId = parentPrefix + $(this).attr("id").substring(childPrefix.length);
    if (parentId) {
      $(`#${parentId} .far.fa-folder`).attr("class", "far fa-folder-open fa-fw");
      $(`#${parentId} i.fas`).removeClass("rotate");
      $(`#${parentId}`).addClass("hide-border-bottom");
    }
  });

});
