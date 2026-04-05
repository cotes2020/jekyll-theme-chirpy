/**
 * 터미널 로그 앞에 ISO 타임스탬프를 붙입니다.
 * 끄려면 환경 변수 `YAWNBOT_CONSOLE_TIMESTAMPS=0` (또는 `false` / `off`).
 */
const raw = process.env.YAWNBOT_CONSOLE_TIMESTAMPS;
const disabled =
  raw !== undefined && ['0', 'false', 'off', 'no'].includes(String(raw).trim().toLowerCase());

if (!disabled) {
  const stamp = () => new Date().toISOString();
  const wrap = (method: 'log' | 'info' | 'warn' | 'error' | 'debug') => {
    const orig = console[method].bind(console) as (...args: unknown[]) => void;
    (console[method] as (...args: unknown[]) => void) = (...args: unknown[]) => {
      orig(`[${stamp()}]`, ...args);
    };
  };
  wrap('log');
  wrap('info');
  wrap('warn');
  wrap('error');
  if (typeof console.debug === 'function') {
    wrap('debug');
  }
}
