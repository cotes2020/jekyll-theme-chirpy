/// <reference types="vite/client" />

interface Window {
  Toolbox?: {
    showToast?: (message: string, variant?: string) => void;
  };
}
