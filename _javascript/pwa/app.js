/* PWA loader */

if ('serviceWorker' in navigator) {
  const meta = document.querySelector('meta[name="pwa-cache"]');
  const isEnabled = meta.content === 'true';

  if (isEnabled) {
    let swUrl = '/sw.min.js';
    const baseUrl = meta.getAttribute('data-baseurl');

    if (baseUrl !== null) {
      swUrl = `${baseUrl}${swUrl}?baseurl=${encodeURIComponent(baseUrl)}`;
    }

    const $notification = $('#notification');
    const $btnRefresh = $('#notification .toast-body>button');

    navigator.serviceWorker.register(swUrl).then((registration) => {
      // In case the user ignores the notification
      if (registration.waiting) {
        $notification.toast('show');
      }

      registration.addEventListener('updatefound', () => {
        registration.installing.addEventListener('statechange', () => {
          if (registration.waiting) {
            if (navigator.serviceWorker.controller) {
              $notification.toast('show');
            }
          }
        });
      });

      $btnRefresh.on('click', () => {
        if (registration.waiting) {
          registration.waiting.postMessage('SKIP_WAITING');
        }
        $notification.toast('hide');
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
