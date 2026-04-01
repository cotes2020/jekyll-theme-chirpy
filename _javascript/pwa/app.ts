import Toast from 'bootstrap/js/src/toast';

if ('serviceWorker' in navigator) {
  const script = document.currentScript;
  if (!(script instanceof HTMLScriptElement)) {
    throw new Error('PWA bootstrap: expected HTMLScriptElement as currentScript');
  }

  const src = new URL(script.src);
  const register = src.searchParams.get('register');
  const baseUrl = src.searchParams.get('baseurl');

  if (register) {
    const swUrl = `${baseUrl}/sw.min.js`;
    const notification = document.getElementById('notification');
    if (!notification) {
      throw new Error('PWA bootstrap: missing #notification element');
    }
    const btnRefresh = notification.querySelector<HTMLButtonElement>(
      '.toast-body>button'
    );
    if (!btnRefresh) {
      throw new Error('PWA bootstrap: missing refresh button in notification toast');
    }
    const popupWindow = Toast.getOrCreateInstance(notification);

    navigator.serviceWorker.register(swUrl).then((registration) => {
      if (registration.waiting) {
        popupWindow.show();
      }

      registration.addEventListener('updatefound', () => {
        registration.installing?.addEventListener('statechange', () => {
          if (registration.waiting && navigator.serviceWorker.controller) {
            popupWindow.show();
          }
        });
      });

      btnRefresh.addEventListener('click', () => {
        if (registration.waiting) {
          registration.waiting.postMessage('SKIP_WAITING');
        }
        popupWindow.hide();
      });
    });

    let refreshing = false;

    navigator.serviceWorker.addEventListener('controllerchange', () => {
      if (!refreshing) {
        window.location.reload();
        refreshing = true;
      }
    });
  } else {
    navigator.serviceWorker.getRegistrations().then((registrations) => {
      for (const registration of registrations) {
        void registration.unregister();
      }
    });
  }
}
