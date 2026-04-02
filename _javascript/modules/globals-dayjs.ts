import { requiredGlobal, type DayjsGlobal } from './globals';

export const dayjs = requiredGlobal<DayjsGlobal>('dayjs');

export function dayjsLocalizedFormatPlugin(): unknown {
  return (window as Window & { dayjs_plugin_localizedFormat?: unknown })
    .dayjs_plugin_localizedFormat;
}
