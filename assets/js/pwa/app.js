---
layout: compress
permalink: '/app.js'
---

const keyWaiting = 'sw-waiting';
const $notification = $('#notification');
const $btnRefresh = $('#notification .toast-body>button');

function skipWating(registration) {
  registration.waiting.postMessage('SKIP_WAITING');
  localStorage.removeItem(keyWaiting);
}

if ('serviceWorker' in navigator) {
  /* Registering Service Worker */
  navigator.serviceWorker.register('{{ "/sw.js" | relative_url }}')
    .then(registration => {
      if (registration) {
        registration.addEventListener('updatefound', () => {
          let serviceWorker = registration.installing;

          serviceWorker.addEventListener('statechange', () => {
            if (serviceWorker.state === 'installed') {
              if (navigator.serviceWorker.controller) {
                $notification.toast('show');
                /* in case the user ignores the notification */
                localStorage.setItem(keyWaiting, true);
              }
            }
          });
        });

        $btnRefresh.click(() => {
          skipWating(registration);
          $notification.toast('hide');
        });

        if (localStorage.getItem(keyWaiting)) {
          if (registration.waiting) {
            /* there's a new Service Worker waiting to be activated */
            $notification.toast('show');
          } else {
            /* closed all open pages after receiving notification */
            localStorage.removeItem(keyWaiting);
          }
        }
      }
    });

  let refreshing = false;

  /* Detect controller change and refresh all the opened tabs */
  navigator.serviceWorker.addEventListener('controllerchange', () => {
    if (!refreshing) {
      window.location.reload();
      refreshing = true;
    }
  });
}
