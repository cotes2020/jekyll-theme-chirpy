// @ts-nocheck
/**
 * imagegen - 설정 상수
 */
(function () {
    'use strict';
    window.ImageGen = window.ImageGen || {};

    const GALLERY_SESSION_KEY = 'toolbox_imagegen_gallery';
    const GALLERY_SESSION_MAX = 5;
    const PROMPT_HISTORY_KEY = 'toolbox_ig_prompt_history';
    const PROMPT_HISTORY_MAX = 20;

    Object.assign(window.ImageGen, {
        GALLERY_SESSION_KEY,
        GALLERY_SESSION_MAX,
        PROMPT_HISTORY_KEY,
        PROMPT_HISTORY_MAX
    });
})();
