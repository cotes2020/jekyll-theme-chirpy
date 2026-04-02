import { requiredGlobal, type GLightboxGlobal } from './globals';

export type { GLightboxInstance } from './globals';

function getGLightbox(): GLightboxGlobal {
  return requiredGlobal<GLightboxGlobal>('GLightbox');
}

/** Resolve window.GLightbox only when opening a lightbox (script not on every layout). */
export function GLightbox(options: { selector: string }) {
  return getGLightbox()(options);
}
