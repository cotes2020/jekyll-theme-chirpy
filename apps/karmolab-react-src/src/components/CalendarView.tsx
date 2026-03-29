import { useState, useCallback, useEffect, useRef } from 'react';
import { Calendar, dateFnsLocalizer, Views } from 'react-big-calendar';
import type { View } from 'react-big-calendar';
import * as DnDImport from 'react-big-calendar/lib/addons/dragAndDrop';
import 'react-big-calendar/lib/css/react-big-calendar.css';
import 'react-big-calendar/lib/addons/dragAndDrop/styles.css';
import {
  format, parse, startOfWeek, getDay, addHours,
  startOfMonth, endOfMonth, eachDayOfInterval, getDate,
  isSameMonth, isSameDay, addMonths, subMonths, subDays,
} from 'date-fns';
import { ko } from 'date-fns/locale';

const extractFunction = (d: any): any => {
  if (typeof d === 'function') return d;
  if (d && typeof d.default === 'function') return d.default;
  if (d && d.default && typeof d.default.default === 'function') return d.default.default;
  return d;
};

const withDragAndDrop = extractFunction(DnDImport);
const DnDCalendar = withDragAndDrop ? withDragAndDrop(Calendar) : Calendar;

const locales = { 'ko': ko };
const localizer = dateFnsLocalizer({ format, parse, startOfWeek, getDay, locales });

const GOOGLE_COLORS: Record<string, string> = {
  '1': '#ac725e', '2': '#d06b64', '3': '#f83a22', '4': '#fa573c',
  '5': '#ff7537', '6': '#ffad46', '7': '#42d692', '8': '#16a765',
  '9': '#7bd148', '10': '#b3dc6c', '11': '#fbe983', '12': '#fad165',
  '13': '#92e1c0', '14': '#9fe1e7', '15': '#9fc6e7', '16': '#4986e7',
  '17': '#9a9cff', '18': '#b99aff', '19': '#c2c2c2', '20': '#cabdbf',
  '21': '#cca6ac', '22': '#f691b2', '23': '#cd74e6', '24': '#a47ae2',
};

function calendarColor(calId: string, googleColor?: string): string {
  if (googleColor) return googleColor;
  const palette = ['#4285f4', '#0f9d58', '#db4437', '#f4b400', '#ab47bc', '#00acc1', '#ff7043', '#43a047'];
  let hash = 0;
  for (let i = 0; i < calId.length; i++) hash = (hash * 31 + calId.charCodeAt(i)) & 0xffffffff;
  return palette[Math.abs(hash) % palette.length];
}

interface GoogleCalendar {
  id: string;
  summary: string;
  backgroundColor?: string;
  selected?: boolean;
}

interface GoogleEvent {
  id: string;
  summary: string;
  start: { dateTime?: string; date?: string; };
  end: { dateTime?: string; date?: string; };
  htmlLink: string;
  colorId?: string;
  calendarId?: string;
}

interface CalendarEvent {
  id: string;
  title: string;
  start: Date;
  end: Date;
  allDay: boolean;
  resource: GoogleEvent;
  color: string;
  calendarName?: string;
}

interface CalendarViewProps {
  accessToken: string;
}

interface EventPopover {
  event: CalendarEvent;
  x: number;
  y: number;
}

interface CreateModal {
  start: Date;
  end: Date;
  allDay: boolean;
}

// --- Mini Calendar Sidebar ---
function MiniCalendar({
  selectedDate,
  onNavigate,
  events,
}: {
  selectedDate: Date;
  onNavigate: (date: Date) => void;
  events: CalendarEvent[];
}) {
  const [miniMonth, setMiniMonth] = useState(new Date(selectedDate));

  useEffect(() => {
    setMiniMonth(new Date(selectedDate));
  }, [selectedDate.getMonth(), selectedDate.getFullYear()]);

  const days = eachDayOfInterval({
    start: startOfWeek(startOfMonth(miniMonth), { locale: ko }),
    end: endOfMonth(miniMonth),
  });
  // Fill to 6 weeks max
  while (days.length % 7 !== 0) {
    days.push(new Date(days[days.length - 1].getTime() + 86400000));
  }

  const today = new Date();
  const dayLabels = ['일', '월', '화', '수', '목', '금', '토'];

  const hasEvent = (day: Date) => events.some(e =>
    isSameDay(e.start, day) || (e.allDay && e.start <= day && e.end >= day)
  );

  return (
    <div className="mini-cal">
      <div className="mini-cal-header">
        <button className="mini-cal-nav" onClick={() => setMiniMonth(d => subMonths(d, 1))}>◀</button>
        <span className="mini-cal-title">{format(miniMonth, 'yyyy년 M월')}</span>
        <button className="mini-cal-nav" onClick={() => setMiniMonth(d => addMonths(d, 1))}>▶</button>
      </div>
      <div className="mini-cal-grid">
        {dayLabels.map(d => (
          <div key={d} className={`mini-cal-dayname ${d === '일' ? 'sun' : d === '토' ? 'sat' : ''}`}>{d}</div>
        ))}
        {days.map((day, i) => {
          const isSelected = isSameDay(day, selectedDate);
          const isToday = isSameDay(day, today);
          const inMonth = isSameMonth(day, miniMonth);
          const dotDay = i % 7 === 0 ? 'sun' : i % 7 === 6 ? 'sat' : '';
          return (
            <button
              key={day.toISOString()}
              className={`mini-cal-day ${isSelected ? 'selected' : ''} ${isToday ? 'today' : ''} ${!inMonth ? 'other-month' : ''} ${dotDay}`}
              onClick={() => onNavigate(day)}
            >
              {getDate(day)}
              {hasEvent(day) && inMonth && <span className="mini-cal-dot" />}
            </button>
          );
        })}
      </div>
    </div>
  );
}

// --- Calendar List ---
function CalendarList({
  calendars,
  hidden,
  onToggle,
}: {
  calendars: GoogleCalendar[];
  hidden: Set<string>;
  onToggle: (id: string) => void;
}) {
  const myCalendars = calendars.filter(c => !c.id.includes('#'));
  const otherCalendars = calendars.filter(c => c.id.includes('#'));

  const Cal = ({ c }: { c: GoogleCalendar }) => {
    const color = calendarColor(c.id, c.backgroundColor);
    const visible = !hidden.has(c.id);
    return (
      <label key={c.id} className="cal-list-item">
        <input
          type="checkbox"
          checked={visible}
          onChange={() => onToggle(c.id)}
          style={{ accentColor: color }}
        />
        <span className="cal-list-dot" style={{ backgroundColor: color }} />
        <span className="cal-list-name">{c.summary}</span>
      </label>
    );
  };

  return (
    <div className="cal-list">
      {myCalendars.length > 0 && (
        <>
          <div className="cal-list-section">내 캘린더</div>
          {myCalendars.map(c => <Cal key={c.id} c={c} />)}
        </>
      )}
      {otherCalendars.length > 0 && (
        <>
          <div className="cal-list-section">다른 캘린더</div>
          {otherCalendars.map(c => <Cal key={c.id} c={c} />)}
        </>
      )}
    </div>
  );
}

// --- Event Modal (Create & Edit) ---
function EventModal({ modal, event, calendars, onClose, onSave }: {
  modal?: CreateModal;
  event?: CalendarEvent;
  calendars: GoogleCalendar[];
  onClose: () => void;
  onSave: (title: string, start: Date, end: Date, allDay: boolean, calendarId: string, eventId?: string) => void;
}) {
  const isEdit = !!event;
  const [title, setTitle] = useState(event?.title || '');
  const [startStr, setStartStr] = useState(format(event?.start || modal?.start || new Date(), "yyyy-MM-dd'T'HH:mm"));
  const [endStr, setEndStr] = useState(format(event?.end || modal?.end || addHours(new Date(), 1), "yyyy-MM-dd'T'HH:mm"));
  const [allDay, setAllDay] = useState(event?.allDay || modal?.allDay || false);
  const [startDate, setStartDate] = useState(format(event?.start || modal?.start || new Date(), 'yyyy-MM-dd'));
  const [calendarId, setCalendarId] = useState(event?.resource.calendarId || 'primary');
  const inputRef = useRef<HTMLInputElement>(null);
  
  useEffect(() => { inputRef.current?.focus(); }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!title.trim()) return;
    let start: Date, end: Date;
    if (allDay) {
      start = new Date(startDate + 'T00:00:00');
      end = new Date(startDate + 'T00:00:00');
    } else {
      start = new Date(startStr);
      end = new Date(endStr);
    }
    onSave(title.trim(), start, end, allDay, calendarId, event?.resource.id);
  };

  return (
    <div className="cal-modal-overlay" onClick={onClose}>
      <div className="cal-modal" onClick={e => e.stopPropagation()}>
        <div className="cal-modal-header">
          <h3 className="cal-modal-title">{isEdit ? '일정 수정' : '새 일정'}</h3>
          <button className="cal-modal-close" onClick={onClose}>✕</button>
        </div>
        <form onSubmit={handleSubmit} className="cal-modal-body">
          <input
            ref={inputRef}
            type="text"
            className="cal-modal-input cal-modal-title-input"
            placeholder="제목 추가"
            value={title}
            onChange={e => setTitle(e.target.value)}
          />
          {calendars.length > 1 && (
            <div className="cal-modal-field">
              <label className="cal-modal-label">캘린더</label>
              <select className="cal-modal-input" value={calendarId} disabled={isEdit} onChange={e => setCalendarId(e.target.value)}>
                {calendars.map(c => <option key={c.id} value={c.id}>{c.summary}</option>)}
              </select>
            </div>
          )}
          <label className="cal-modal-check-label">
            <input type="checkbox" checked={allDay} onChange={e => setAllDay(e.target.checked)} />
            <span>종일</span>
          </label>
          {allDay ? (
            <div className="cal-modal-field">
              <label className="cal-modal-label">날짜</label>
              <input type="date" className="cal-modal-input" value={startDate} onChange={e => setStartDate(e.target.value)} />
            </div>
          ) : (
            <div className="cal-modal-times">
              <div className="cal-modal-field">
                <label className="cal-modal-label">시작</label>
                <input type="datetime-local" className="cal-modal-input" value={startStr} onChange={e => setStartStr(e.target.value)} />
              </div>
              <div className="cal-modal-field">
                <label className="cal-modal-label">종료</label>
                <input type="datetime-local" className="cal-modal-input" value={endStr} onChange={e => setEndStr(e.target.value)} />
              </div>
            </div>
          )}
          <div className="cal-modal-actions">
            <button type="button" className="btn btn-ghost" onClick={onClose}>취소</button>
            <button type="submit" className="btn btn-accent" disabled={!title.trim()}>저장</button>
          </div>
        </form>
      </div>
    </div>
  );
}

// --- Event Popover ---
function EventPopoverUI({ popover, onClose, onEdit, onDelete }: {
  popover: EventPopover;
  onClose: () => void;
  onEdit: (event: CalendarEvent) => void;
  onDelete: (event: CalendarEvent) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const { event } = popover;
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose();
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [onClose]);

  const timeStr = event.allDay
    ? format(event.start, 'yyyy년 M월 d일 (EEE)', { locale: ko })
    : `${format(event.start, 'M월 d일 (EEE) HH:mm', { locale: ko })} – ${format(event.end, 'HH:mm')}`;

  const left = Math.min(popover.x, window.innerWidth - 300);
  const top = Math.min(popover.y, window.innerHeight - 220);

  return (
    <div ref={ref} className="cal-popover" style={{ left, top }}>
      <div className="cal-popover-header">
        <div className="cal-popover-dot" style={{ backgroundColor: event.color }} />
        <button className="cal-modal-close" onClick={onClose}>✕</button>
      </div>
      <div className="cal-popover-title">{event.title}</div>
      <div className="cal-popover-time">🕐 {timeStr}</div>
      {event.calendarName && (
        <div className="cal-popover-cal">📅 {event.calendarName}</div>
      )}
      <div className="cal-popover-actions">
        <button className="btn btn-ghost cal-popover-btn" onClick={() => onEdit(event)}>기본 수정</button>
        <a href={event.resource.htmlLink} target="_blank" rel="noopener noreferrer" className="btn btn-ghost cal-popover-btn">
          Google ↗
        </a>
        <button className="btn btn-danger cal-popover-btn" onClick={() => onDelete(event)}>삭제</button>
      </div>
    </div>
  );
}


// --- Main CalendarView ---
export function CalendarView({ accessToken }: CalendarViewProps) {
  const [events, setEvents] = useState<CalendarEvent[]>([]);
  const [calendars, setCalendars] = useState<GoogleCalendar[]>([]);
  const [hiddenCalendars, setHiddenCalendars] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(false);
  const [currentDate, setCurrentDate] = useState(new Date());
  const [currentView, setCurrentView] = useState<View>(Views.WEEK);
  const [createModal, setCreateModal] = useState<CreateModal | null>(null);
  const [editingEvent, setEditingEvent] = useState<CalendarEvent | null>(null);
  const [popover, setPopover] = useState<EventPopover | null>(null);

  const fetchCalendars = useCallback(async () => {
    if (!accessToken) return;
    try {
      const res = await fetch('https://www.googleapis.com/calendar/v3/users/me/calendarList', {
        headers: { Authorization: `Bearer ${accessToken}` },
      });
      const data = await res.json();
      if (data.items) setCalendars(data.items);
    } catch (e) {
      console.error('Failed to fetch calendars', e);
    }
  }, [accessToken]);

  useEffect(() => { fetchCalendars(); }, [fetchCalendars]);

  const fetchEvents = useCallback(async (start: Date, end: Date, calList: GoogleCalendar[]) => {
    if (!accessToken) return;
    setLoading(true);
    try {
      const targetCals = calList.length > 0 ? calList : [{ id: 'primary', summary: '기본' }];
      const results = await Promise.all(targetCals.map(async (cal) => {
        const params = new URLSearchParams({
          timeMin: start.toISOString(),
          timeMax: end.toISOString(),
          singleEvents: 'true',
          orderBy: 'startTime',
        });
        const res = await fetch(`https://www.googleapis.com/calendar/v3/calendars/${encodeURIComponent(cal.id)}/events?${params}`, {
          headers: { Authorization: `Bearer ${accessToken}` },
        });
        const data = await res.json();
        return { cal, items: data.items || [] };
      }));

      const allEvents: CalendarEvent[] = [];
      results.forEach(({ cal, items }) => {
        const baseColor = calendarColor(cal.id, cal.backgroundColor);
        items.forEach((item: GoogleEvent) => {
          const color = item.colorId ? (GOOGLE_COLORS[item.colorId] || baseColor) : baseColor;
          const isAllDay = !!item.start.date;
          const startDate = new Date(item.start.dateTime || item.start.date || new Date());
          let endDate = new Date(item.end.dateTime || item.end.date || new Date());
          // Fix: Google's allDay end.date is EXCLUSIVE (next day). Subtract 1 day for display.
          if (isAllDay) {
            endDate = subDays(endDate, 1);
          }
          allEvents.push({
            id: `${cal.id}_${item.id}`,
            title: item.summary || '(제목 없음)',
            start: startDate,
            end: endDate,
            allDay: isAllDay,
            resource: { ...item, calendarId: cal.id },
            color,
            calendarName: cal.summary,
          });
        });
      });
      setEvents(allEvents);
    } catch (e) {
      console.error('Failed to fetch calendar events', e);
    } finally {
      setLoading(false);
    }
  }, [accessToken]);

  const refetchCurrent = useCallback(() => {
    const start = new Date(currentDate.getFullYear(), currentDate.getMonth() - 1, 1);
    const end = new Date(currentDate.getFullYear(), currentDate.getMonth() + 2, 0);
    fetchEvents(start, end, calendars);
  }, [currentDate, fetchEvents, calendars]);

  useEffect(() => { refetchCurrent(); }, [calendars, currentDate]); // eslint-disable-line

  const visibleEvents = events.filter(e => !hiddenCalendars.has(e.resource.calendarId || ''));

  const toggleCalendar = (id: string) => {
    setHiddenCalendars(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  };

  const onEventDrop = async ({ event, start, end }: any) => {
    if (!accessToken) return;
    const ce = event as CalendarEvent;
    setEvents(prev => prev.map(e => e.id === ce.id ? { ...e, start, end } : e));
    const calId = ce.resource.calendarId || 'primary';
    // When writing back, all-day events need end date +1 for Google API
    const payload = ce.allDay
      ? { start: { date: format(start, 'yyyy-MM-dd') }, end: { date: format(addHours(end, 24), 'yyyy-MM-dd') } }
      : { start: { dateTime: start.toISOString() }, end: { dateTime: end.toISOString() } };
    await fetch(`https://www.googleapis.com/calendar/v3/calendars/${encodeURIComponent(calId)}/events/${ce.resource.id}`, {
      method: 'PATCH',
      headers: { Authorization: `Bearer ${accessToken}`, 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  };

  const onSelectSlot = (slotInfo: { start: Date; end: Date; action: string }) => {
    if (!accessToken) return;
    setPopover(null);
    const isAllDay = slotInfo.action === 'select' && slotInfo.start.getHours() === 0 && slotInfo.end.getHours() === 0;
    const end = isAllDay ? slotInfo.end : (slotInfo.end.getTime() === slotInfo.start.getTime() ? addHours(slotInfo.start, 1) : slotInfo.end);
    setCreateModal({ start: slotInfo.start, end, allDay: isAllDay });
  };

  const handleSaveEvent = async (title: string, start: Date, end: Date, allDay: boolean, calId: string, eventId?: string) => {
    setCreateModal(null);
    setEditingEvent(null);
    try {
      // For all-day events, end date must be +1 for Google API
      const payload = allDay
        ? { summary: title, start: { date: format(start, 'yyyy-MM-dd') }, end: { date: format(addHours(end, 24), 'yyyy-MM-dd') } }
        : { summary: title, start: { dateTime: start.toISOString() }, end: { dateTime: end.toISOString() } };
      const url = eventId 
        ? `https://www.googleapis.com/calendar/v3/calendars/${encodeURIComponent(calId)}/events/${eventId}`
        : `https://www.googleapis.com/calendar/v3/calendars/${encodeURIComponent(calId)}/events`;
      
      await fetch(url, {
        method: eventId ? 'PATCH' : 'POST',
        headers: { Authorization: `Bearer ${accessToken}`, 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      setTimeout(() => refetchCurrent(), 500);
    } catch (e) {
      console.error('Failed to create event', e);
    }
  };

  const handleSelectEvent = (event: object, e: React.SyntheticEvent) => {
    const ce = event as CalendarEvent;
    const rect = (e.target as HTMLElement).getBoundingClientRect();
    setCreateModal(null);
    setEditingEvent(null);
    setPopover({ event: ce, x: rect.left, y: rect.bottom + 8 });
  };

  const handleEditEvent = (event: CalendarEvent) => {
    setPopover(null);
    setEditingEvent(event);
  };

  const handleDeleteEvent = async (event: CalendarEvent) => {
    if (!confirm(`"${event.title}" 일정을 삭제하시겠습니까?`)) return;
    setPopover(null);
    const calId = event.resource.calendarId || 'primary';
    try {
      await fetch(`https://www.googleapis.com/calendar/v3/calendars/${encodeURIComponent(calId)}/events/${event.resource.id}`, {
        method: 'DELETE',
        headers: { Authorization: `Bearer ${accessToken}` },
      });
      setEvents(prev => prev.filter(e => e.id !== event.id));
    } catch (e) {
      console.error('Failed to delete event', e);
    }
  };

  const eventStyleGetter = (event: object) => {
    const ce = event as CalendarEvent;
    return {
      style: {
        backgroundColor: ce.color,
        borderRadius: '4px',
        color: '#ffffff',
        border: 'none',
        fontSize: '12px',
        fontWeight: 600,
        padding: '2px 6px',
        boxShadow: `0 1px 4px ${ce.color}66`,
      }
    };
  };

  return (
    <div className="calendar-layout">
      {/* Left Sidebar */}
      <div className="calendar-sidebar">
        <button
          className="cal-create-btn"
          onClick={() => { setEditingEvent(null); setCreateModal({ start: new Date(), end: addHours(new Date(), 1), allDay: false }); }}
        >
          + 만들기
        </button>
        <MiniCalendar selectedDate={currentDate} onNavigate={d => { setCurrentDate(d); setCurrentView(Views.DAY); }} events={visibleEvents} />
        <CalendarList calendars={calendars} hidden={hiddenCalendars} onToggle={toggleCalendar} />
      </div>

      {/* Main Calendar Area */}
      <div className="calendar-wrapper">
        {loading && <div className="calendar-loading-overlay">⏳ 불러오는 중...</div>}
        <DnDCalendar
          localizer={localizer}
          events={visibleEvents}
          startAccessor={(event: CalendarEvent) => event.start}
          endAccessor={(event: CalendarEvent) => event.end}
          views={[Views.MONTH, Views.WEEK, Views.DAY]}
          view={currentView}
          date={currentDate}
          onNavigate={setCurrentDate}
          onView={setCurrentView}
          onSelectEvent={handleSelectEvent}
          selectable={true}
          onSelectSlot={onSelectSlot}
          onEventDrop={onEventDrop}
          onEventResize={onEventDrop}
          resizable={true}
          eventPropGetter={eventStyleGetter}
          culture="ko"
          getNow={() => new Date()}
          scrollToTime={new Date(1970, 1, 1, 8, 0, 0)}
          messages={{
            today: '오늘',
            previous: '◀',
            next: '▶',
            month: '월간',
            week: '주간',
            day: '일간',
            agenda: '목록',
            date: '날짜',
            time: '시간',
            event: '일정',
            noEventsInRange: '이 기간에는 일정이 없습니다.',
            showMore: (total: number) => `+${total}개 더보기`,
          }}
        />
      </div>

      {(createModal || editingEvent) && (
        <EventModal
          modal={createModal || undefined}
          event={editingEvent || undefined}
          calendars={calendars}
          onClose={() => { setCreateModal(null); setEditingEvent(null); }}
          onSave={handleSaveEvent}
        />
      )}
      {popover && (
        <EventPopoverUI
          popover={popover}
          onClose={() => setPopover(null)}
          onEdit={handleEditEvent}
          onDelete={handleDeleteEvent}
        />
      )}
    </div>
  );
}
