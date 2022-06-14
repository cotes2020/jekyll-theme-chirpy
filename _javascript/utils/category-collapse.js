/*
 * Tab 'Categories' expand/close effect.
 */

$(function () {
  const childPrefix = "l_";
  const parentPrefix = "h_";
  const collapse = $(".collapse");

  /* close up top-category */
  collapse.on("hide.bs.collapse", function () { /* Bootstrap collapse events. */
    const parentId = parentPrefix + $(this).attr("id").substring(childPrefix.length);
    if (parentId) {
      $(`#${parentId} .bi.bi-folder2-open`).attr("class", "bi bi-folder-fill");
      $(`#${parentId} i.bi`).addClass("rotate");
      $(`#${parentId}`).removeClass("hide-border-bottom");
    }
  });

  /* expand the top category */
  collapse.on("show.bs.collapse", function () {
    const parentId = parentPrefix + $(this).attr("id").substring(childPrefix.length);
    if (parentId) {
      $(`#${parentId} .bi.bi-folder-fill`).attr("class", "bi bi-folder2-open");
      $(`#${parentId} i.bi`).removeClass("rotate");
      $(`#${parentId}`).addClass("hide-border-bottom");
    }
  });

});
