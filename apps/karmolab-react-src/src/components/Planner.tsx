import { useState, lazy, Suspense } from 'react';

const CalendarView = lazy(() =>
  import('./CalendarView').then((m) => ({ default: m.CalendarView }))
);
const KanbanView = lazy(() =>
  import('./KanbanView').then((m) => ({ default: m.KanbanView }))
);

interface PlannerProps {
  accessToken: string;
}

export function Planner({ accessToken }: PlannerProps) {
  const [activeTab, setActiveTab] = useState<'calendar' | 'tasks'>('calendar');

  return (
    <div className="planner-dashboard">
      <div className="planner-tab-bar">
        <button
          className={`planner-tab-btn ${activeTab === 'calendar' ? 'active' : ''}`}
          onClick={() => setActiveTab('calendar')}
        >
          🗓 캘린더
        </button>
        <button
          className={`planner-tab-btn ${activeTab === 'tasks' ? 'active' : ''}`}
          onClick={() => setActiveTab('tasks')}
        >
          📋 칸반
        </button>
      </div>

      <div className="planner-content">
        <Suspense fallback={<p className="planner-tab-loading">탭을 불러오는 중…</p>}>
          {activeTab === 'calendar' && <CalendarView accessToken={accessToken} />}
          {activeTab === 'tasks' && <KanbanView accessToken={accessToken} />}
        </Suspense>
      </div>
    </div>
  );
}
