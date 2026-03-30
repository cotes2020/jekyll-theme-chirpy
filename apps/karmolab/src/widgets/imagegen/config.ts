/**
 * imagegen - 설정 상수
 */
(function (): void {
  'use strict';

  const GALLERY_SESSION_KEY = 'toolbox_imagegen_gallery';
  const GALLERY_SESSION_MAX = 5;
  const PROMPT_HISTORY_KEY = 'toolbox_ig_prompt_history';
  const PROMPT_HISTORY_MAX = 20;

  window.ImageGen = window.ImageGen || ({} as NonNullable<Window['ImageGen']>);
  Object.assign(window.ImageGen, {
    GALLERY_SESSION_KEY,
    GALLERY_SESSION_MAX,
    PROMPT_HISTORY_KEY,
    PROMPT_HISTORY_MAX
  });
})();
