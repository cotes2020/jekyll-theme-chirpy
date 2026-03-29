import { useState } from 'react';
import { CalendarView } from './CalendarView';
import { KanbanView } from './KanbanView';

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
        {activeTab === 'calendar' && <CalendarView accessToken={accessToken} />}
        {activeTab === 'tasks' && <KanbanView accessToken={accessToken} />}
      </div>
    </div>
  );
}
