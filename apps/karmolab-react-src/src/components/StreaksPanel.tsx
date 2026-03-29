import { Flame } from 'lucide-react';
import { useCallback, useState } from 'react';
import {
  DEFAULT_TRACKS,
  loadUserData,
  localDateString,
  recordStreakActivity,
} from '../lib/gamification';

export function StreaksPanel() {
  const [version, setVersion] = useState(0);
  const refresh = useCallback(() => setVersion((v) => v + 1), []);

  const data = loadUserData();
  const today = localDateString();

  return (
    <section className="field-group">
      <div style={{ marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '12px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', width: '40px', height: '40px', borderRadius: 'var(--radius-md)', background: 'var(--warning-subtle)', color: 'var(--warning)' }}>
          <Flame style={{ width: '24px', height: '24px' }} aria-hidden />
        </div>
        <div>
          <h2 className="field-label" style={{ marginBottom: '4px' }}>스트릭</h2>
          <p style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-tertiary)' }}>오늘 기준: {today}</p>
        </div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }} key={version}>
        {DEFAULT_TRACKS.map((track) => {
          const s = data.streaks[track.id];
          const current = s?.current ?? 0;
          const longest = s?.longest ?? 0;
          const last = s?.lastActivityDate ?? '—';
          const doneToday = s?.lastActivityDate === today;

          return (
            <div
              key={track.id}
              className="result-header"
              style={{ borderRadius: 'var(--radius-md)', padding: '12px 16px', display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: '16px' }}
            >
              <div>
                <p style={{ fontWeight: 600, color: 'var(--text-secondary)' }}>{track.label}</p>
                <p style={{ fontSize: '12px', color: 'var(--text-tertiary)', marginTop: '4px' }}>
                  현재 <span style={{ color: 'var(--warning)' }}>{current}</span>일 · 최장{' '}
                  <span style={{ color: 'var(--text-secondary)' }}>{longest}</span>일 · 마지막 {last}
                </p>
              </div>
              <button
                type="button"
                disabled={doneToday}
                onClick={() => {
                  recordStreakActivity(track.id);
                  refresh();
                }}
                className={`btn ${doneToday ? 'btn-ghost' : 'btn-accent'}`}
                style={doneToday ? { cursor: 'default' } : {}}
              >
                {doneToday ? '오늘 완료됨' : '오늘 완료'}
              </button>
            </div>
          );
        })}
      </div>

      <p style={{ marginTop: '16px', textAlign: 'center', fontSize: '12px', color: 'var(--text-tertiary)' }}>
        캘린더·칸반 연동 시 같은 기록 API로 자동 반영됩니다.
      </p>
    </section>
  );
}
