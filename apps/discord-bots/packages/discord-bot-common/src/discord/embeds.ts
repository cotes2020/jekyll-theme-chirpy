/** Discord embed 길이 제한 유틸 */

export function truncateEmbedField(str: unknown, max = 1000): string {
  if (str == null || str === '') return '';
  const t = String(str);
  if (t.length <= max) return t;
  return `${t.slice(0, max - 1)}…`;
}

export function truncateDiscordDescription(str: unknown, max = 4090): string {
  if (str == null || str === '') return '';
  const t = String(str);
  if (t.length <= max) return t;
  return `${t.slice(0, max - 1)}…`;
}

export function hasGitWorkingChanges(g: any): boolean {
  if (!g || !g.isRepo) return false;
  const st = String(g.statusPorcelain || '').trim();
  const ds = String(g.diffStat || '').trim();
  const dp = String(g.diffPreview || '').trim();
  return st.length > 0 || ds.length > 0 || dp.length > 0;
}

export function formatCursorAcpRpcSummaryField(s: any): string {
  const q = typeof s?.askQuestionCount === 'number' ? s.askQuestionCount : 0;
  const p = typeof s?.createPlanCount === 'number' ? s.createPlanCount : 0;
  const lineAsk =
    q === 0
      ? '**cursor/ask_question** · 호출 없음 — 디스코드 선택 메뉴는 이 RPC가 있을 때만 뜹니다.'
      : `**cursor/ask_question** · ${q}회 호출됨 (디스코드에서 선택 연동).`;
  const linePlan =
    p === 0 ? '**cursor/create_plan** · 호출 없음.' : `**cursor/create_plan** · ${p}회 요청됨 (러너 자동 승인, 별도 디스코드 UI 없음).`;
  return truncateEmbedField(`${lineAsk}\n${linePlan}`, 1020);
}

