---
layout: compress
# PWA service worker
---

self.importScripts('{{ "/assets/js/data/cache-list.js" | relative_url }}');

var cacheName = 'chirpy-{{ "now" | date: "%Y%m%d.%H%M" }}';


function isExcluded(url) {
  const regex = /(^http(s)?|^\/)/; /* the regex for CORS url or relative url */
  for (const rule of exclude) {
    if (!regex.test(url) ||
      url.indexOf(rule) != -1) {
      return true;
    }
  }
  return false;
}


self.addEventListener('install', (e) => {
  self.skipWaiting();
  e.waitUntil(
    caches.open(cacheName).then((cache) => {
      return cache.addAll(include);
    })
  );
});


self.addEventListener('fetch', (e) => {
  e.respondWith(
    caches.match(e.request).then((r) => {
      /* console.log('[Service Worker] Fetching resource: ' + e.request.url); */
      return r || fetch(e.request).then((response) => {
        return caches.open(cacheName).then((cache) => {
          if (!isExcluded(e.request.url)) {
            /* console.log('[Service Worker] Caching new resource: ' + e.request.url); */
            cache.put(e.request, response.clone());
          }
          return response;
        });
      });
    })
  );
});


self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then((keyList) => {
          return Promise.all(keyList.map((key) => {
        if(key !== cacheName) {
          return caches.delete(key);
        }
      }));
    })
  );
});
