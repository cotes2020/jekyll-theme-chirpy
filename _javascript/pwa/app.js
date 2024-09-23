import Toast from 'bootstrap/js/src/toast';

if ('serviceWorker' in navigator) {
  // Get Jekyll config from URL parameters
  const src = new URL(document.currentScript.src);
  const register = src.searchParams.get('register');
  const baseUrl = src.searchParams.get('baseurl');

  if (register) {
    const swUrl = `${baseUrl}/sw.min.js`;
    const notification = document.getElementById('notification');
    const btnRefresh = notification.querySelector('.toast-body>button');
    const popupWindow = Toast.getOrCreateInstance(notification);

    navigator.serviceWorker.register(swUrl).then((registration) => {
      // Restore the update window that was last manually closed by the user
      if (registration.waiting) {
        popupWindow.show();
      }

      registration.addEventListener('updatefound', () => {
        registration.installing.addEventListener('statechange', () => {
          if (registration.waiting) {
            if (navigator.serviceWorker.controller) {
              popupWindow.show();
            }
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

    // Detect controller change and refresh all the opened tabs
    navigator.serviceWorker.addEventListener('controllerchange', () => {
      if (!refreshing) {
        window.location.reload();
        refreshing = true;
      }
    });
  } else {
    navigator.serviceWorker.getRegistrations().then(function (registrations) {
      for (let registration of registrations) {
        registration.unregister();
      }
    });
  }
}
