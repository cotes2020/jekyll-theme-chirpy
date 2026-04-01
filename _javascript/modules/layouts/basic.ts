import { back2top, loadTooltip, modeWatcher } from '../components';

export function basic(): void {
  modeWatcher();
  back2top();
  loadTooltip();
}
