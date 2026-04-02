import { requiredGlobal, type ClipboardJSGlobal } from './globals';

function getClipboardCtor(): ClipboardJSGlobal {
  return requiredGlobal<ClipboardJSGlobal>('ClipboardJS');
}

/**
 * Resolve window.ClipboardJS only on `new` (script is not on every layout, e.g. home).
 */
export const ClipboardJS = new Proxy(
  function ClipboardJSStub() {
    /* construct trap handles instantiation */
  } as unknown as ClipboardJSGlobal,
  {
    construct(_target, args) {
      const Ctor = getClipboardCtor();
      return Reflect.construct(Ctor, args, Ctor);
    }
  }
) as ClipboardJSGlobal;
