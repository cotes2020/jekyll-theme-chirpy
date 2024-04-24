---
layout: compress
permalink: /assets/js/dist/:basename.min.js
---

if ('serviceWorker' in navigator) {
  const isEnabled = '{{ site.pwa.enabled }}' === 'true';

  if (isEnabled) {
    const swUrl = '{{ '/sw.min.js' | relative_url }}';
    const notification = document.getElementById('notification');
    const btnRefresh = notification.querySelector('.toast-body>button');
    const popupWindow = bootstrap.Toast.getOrCreateInstance(notification);

    navigator.serviceWorker.register(swUrl).then((registration) => {
      {% comment %}In case the user ignores the notification{% endcomment %}
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

    {% comment %}Detect controller change and refresh all the opened tabs{% endcomment %}
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
