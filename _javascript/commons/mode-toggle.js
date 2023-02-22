/**
 * Listener for theme mode toggle
 */
$(function () {
    $(".mode-toggle").on('click',(e) => {
        const $target = $(e.target);
        let $btn = ($target.prop("tagName") === "button".toUpperCase() ?
            $target : $target.parent());

        $btn.trigger('blur'); // remove the clicking outline
        flipMode();
    });
});
