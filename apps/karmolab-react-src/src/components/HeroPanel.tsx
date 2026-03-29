import { useState, useCallback } from 'react';
import { Flame } from 'lucide-react';
import {
    DEFAULT_TRACKS,
    loadUserData,
    localDateString,
    recordStreakActivity,
    getLevelProgress,
    getLevelTitle,
    getLevelRange,
} from '../lib/gamification';

export function HeroPanel() {
    const [version, setVersion] = useState(0);
    const refresh = useCallback(() => setVersion((v) => v + 1), []);

    const data = loadUserData();
    const today = localDateString();

    const totalExp = data.totalExp || 0;
    const level = data.level || 0;
    const progress = getLevelProgress(totalExp);
    const title = getLevelTitle(level);
    const { min, max } = getLevelRange(level);
    const expIntoLevel = totalExp - min;
    const expNeeded = max - min;

    return (
        <section className="field-group hero-panel" key={version}>
            {/* Level Hero Card */}
            <div className="hero-level-card">
                <div className="hero-level-badge">{level}</div>
                <div className="hero-level-info">
                    <div className="hero-level-title">{title}</div>
                    <div className="hero-exp-label">EXP {totalExp.toLocaleString()} — Lv.{level} 구간 ({expIntoLevel}/{expNeeded})</div>
                    <div className="hero-exp-bar-bg">
                        <div
                            className="hero-exp-bar-fill"
                            style={{ width: `${Math.min(progress * 100, 100).toFixed(1)}%` }}
                        />
                    </div>
                </div>
            </div>

            {/* Streaks */}
            <div className="hero-streaks-row">
                {DEFAULT_TRACKS.map((track) => {
                    const s = data.streaks[track.id];
                    const current = s?.current ?? 0;
                    const longest = s?.longest ?? 0;
                    const doneToday = s?.lastActivityDate === today;

                    return (
                        <div key={track.id} className={`hero-streak-card ${doneToday ? 'done' : ''}`}>
                            <div className="hero-streak-flame">
                                <Flame style={{ width: 20, height: 20 }} />
                            </div>
                            <div className="hero-streak-info">
                                <div className="hero-streak-label">{track.label}</div>
                                <div className="hero-streak-stats">
                                    <span className="hero-streak-current">{current}일 연속</span>
                                    <span className="hero-streak-max"> / 최장 {longest}일</span>
                                </div>
                            </div>
                            <button
                                type="button"
                                disabled={doneToday}
                                onClick={() => { recordStreakActivity(track.id); refresh(); }}
                                className={`btn ${doneToday ? 'btn-ghost' : 'btn-accent'} hero-streak-btn`}
                            >
                                {doneToday ? '✅ 완료' : '+30 EXP'}
                            </button>
                        </div>
                    );
                })}
            </div>
        </section>
    );
}
