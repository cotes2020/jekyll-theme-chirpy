import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import type { Root } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

let root: Root | null = null;

(window as any).mountKarmoPlanner = (containerId: string) => {
  const container = document.getElementById(containerId);
  if (!container) return;
  
  if (root) {
    root.unmount();
  }
  
  root = createRoot(container);
  root.render(
    <StrictMode>
      <App />
    </StrictMode>
  );
};

// Auto-mount for development isolation or first-load timing
if (document.getElementById('karmolab-planner-root')) {
  (window as any).mountKarmoPlanner('karmolab-planner-root');
}
