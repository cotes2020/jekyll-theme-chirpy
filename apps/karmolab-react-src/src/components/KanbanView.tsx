import { useState, useCallback, useEffect } from 'react';
import { DragDropContext, Droppable, Draggable } from '@hello-pangea/dnd';
import type { DropResult } from '@hello-pangea/dnd';

interface GoogleTask {
  id: string;
  title: string;
  notes?: string;
  status: 'needsAction' | 'completed';
}

interface KanbanData {
  todo: GoogleTask[];
  inProgress: GoogleTask[];
  done: GoogleTask[];
}

interface KanbanViewProps {
  accessToken: string;
}

const IN_PROGRESS_TAG = '[IN_PROGRESS]';

export function KanbanView({ accessToken }: KanbanViewProps) {
  const [data, setData] = useState<KanbanData>({ todo: [], inProgress: [], done: [] });
  const [loading, setLoading] = useState(false);
  const [addingTask, setAddingTask] = useState(false);
  const [newTaskTitle, setNewTaskTitle] = useState('');

  const fetchTasks = useCallback(async () => {
    if (!accessToken) return;
    setLoading(true);
    try {
      const response = await fetch('https://tasks.googleapis.com/tasks/v1/lists/@default/tasks?showCompleted=true&showHidden=true', {
        headers: { Authorization: `Bearer ${accessToken}` },
      });
      const result = await response.json();
      if (result.items) {
        const todo: GoogleTask[] = [];
        const inProgress: GoogleTask[] = [];
        const done: GoogleTask[] = [];

        result.items.forEach((task: GoogleTask) => {
          if (task.status === 'completed') {
            done.push(task);
          } else if (task.notes && task.notes.includes(IN_PROGRESS_TAG)) {
            inProgress.push(task);
          } else {
            todo.push(task);
          }
        });

        setData({ todo, inProgress, done });
      }
    } catch (e) {
      console.error('Failed to fetch tasks', e);
    } finally {
      setLoading(false);
    }
  }, [accessToken]);

  useEffect(() => {
    fetchTasks();
  }, [fetchTasks]);

  const updateTaskInGoogle = async (taskId: string, payload: any) => {
    try {
      await fetch(`https://tasks.googleapis.com/tasks/v1/lists/@default/tasks/${taskId}`, {
        method: 'PATCH',
        headers: {
          Authorization: `Bearer ${accessToken}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      });
    } catch (e) {
      console.error('Failed to update task', e);
    }
  };

  const handleDragEnd = async (result: DropResult) => {
    const { source, destination } = result;
    if (!destination) return;
    if (source.droppableId === destination.droppableId && source.index === destination.index) return;

    const sourceColId = source.droppableId as keyof KanbanData;
    const destColId = destination.droppableId as keyof KanbanData;

    const sourceList = Array.from(data[sourceColId] || []);
    const destList = sourceColId === destColId ? sourceList : Array.from(data[destColId] || []);
    
    // Optimistic UI update
    const [movedTask] = sourceList.splice(source.index, 1);
    destList.splice(destination.index, 0, movedTask);

    setData(prev => ({
      ...prev,
      [sourceColId]: sourceList,
      [destColId]: destList
    }));

    // Update Google Tasks API
    let newStatus: 'needsAction' | 'completed' = 'needsAction';
    let newNotes = movedTask.notes || '';

    if (destColId === 'done') {
      newStatus = 'completed';
    } else {
      newStatus = 'needsAction';
      if (destColId === 'inProgress') {
        if (!newNotes.includes(IN_PROGRESS_TAG)) {
          newNotes = newNotes ? `${newNotes}\n${IN_PROGRESS_TAG}` : IN_PROGRESS_TAG;
        }
      } else if (destColId === 'todo') {
        newNotes = newNotes.replace(IN_PROGRESS_TAG, '').trim();
      }
    }

    const payload: any = { status: newStatus };
    if (newStatus === 'needsAction') {
      payload.notes = newNotes;
      // Complete state removes completed timestamp automatically via API 
    }

    await updateTaskInGoogle(movedTask.id, payload);
    // Silent refetch to ensure sync
    fetchTasks();
  };

  const handleAddTask = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newTaskTitle.trim() || !accessToken) return;

    setAddingTask(true);
    try {
      const response = await fetch('https://tasks.googleapis.com/tasks/v1/lists/@default/tasks', {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${accessToken}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ title: newTaskTitle }),
      });
      const data = await response.json();
      if (data.id) {
        setNewTaskTitle('');
        fetchTasks();
      }
    } catch (e) {
      console.error('Failed to add task', e);
    } finally {
      setAddingTask(false);
    }
  };

  const columns = [
    { id: 'todo', title: '해야 할 일' },
    { id: 'inProgress', title: '진행 중' },
    { id: 'done', title: '완료' }
  ];

  return (
    <div className="kanban-wrapper">
      <form onSubmit={handleAddTask} className="task-add-form" style={{ marginBottom: '16px' }}>
        <input
          type="text"
          className="task-add-input"
          placeholder="새로운 할 일을 입력하세요..."
          value={newTaskTitle}
          onChange={(e) => setNewTaskTitle(e.target.value)}
          disabled={addingTask}
        />
        <button type="submit" className="btn btn-accent" disabled={addingTask || !newTaskTitle.trim()}>
          {addingTask ? '추가 중...' : '추가'}
        </button>
      </form>

      {loading && <div className="kanban-loading">불러오는 중...</div>}

      <DragDropContext onDragEnd={handleDragEnd}>
        <div className="kanban-board">
          {columns.map(col => (
            <div key={col.id} className="kanban-column">
              <div className="kanban-column-header">{col.title} <span className="kanban-count">{data[col.id as keyof KanbanData]?.length || 0}</span></div>
              <Droppable droppableId={col.id}>
                {(provided, snapshot) => (
                  <div
                    ref={provided.innerRef}
                    {...provided.droppableProps}
                    className={`kanban-column-content ${snapshot.isDraggingOver ? 'dragging-over' : ''}`}
                  >
                    {data[col.id as keyof KanbanData]?.map((task, index) => (
                      <Draggable key={task.id} draggableId={task.id} index={index}>
                        {(provided, snapshot) => (
                          <div
                            ref={provided.innerRef}
                            {...provided.draggableProps}
                            {...provided.dragHandleProps}
                            className={`kanban-card ${snapshot.isDragging ? 'dragging' : ''} ${task.status === 'completed' ? 'completed' : ''}`}
                          >
                            <div className="kanban-card-title">{task.title}</div>
                            {task.notes && task.notes.replace(IN_PROGRESS_TAG, '').trim() && (
                              <div className="kanban-card-notes">{task.notes.replace(IN_PROGRESS_TAG, '').trim()}</div>
                            )}
                          </div>
                        )}
                      </Draggable>
                    ))}
                    {provided.placeholder}
                  </div>
                )}
              </Droppable>
            </div>
          ))}
        </div>
      </DragDropContext>
    </div>
  );
}
