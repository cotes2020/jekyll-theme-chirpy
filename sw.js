---
layout: compress
# PWA service worker
---

self.importScripts('{{ "/assets/js/data/swcache.js" | relative_url }}');

const cacheName = 'chirpy-{{ "now" | date: "%Y%m%d.%H%M" }}';

function verifyDomain(url) {
  for (const domain of allowedDomains) {
    const regex = RegExp(`^http(s)?:\/\/${domain}\/`);
    if (regex.test(url)) {
      return true;
    }
  }

  return false;
}

function isExcluded(url) {
  for (const item of denyUrls) {
    if (url === item) {
      return true;
    }
  }
  return false;
}

self.addEventListener('install', e => {
  self.skipWaiting();
  e.waitUntil(
    caches.open(cacheName).then(cache => {
      return cache.addAll(resource);
    })
  );
});

self.addEventListener('fetch', e => {
  e.respondWith(
    caches.match(e.request).then(r => {
      /* console.log(`[sw] method: ${e.request.method}, fetching: ${e.request.url}`); */
      return r || fetch(e.request).then(response => {
        const url = e.request.url;

        if (e.request.method !== 'GET'
          || !verifyDomain(url)
          || isExcluded(url)) {
          return response;
        }

        return caches.open(cacheName).then(cache => {
          /* console.log('[sw] Caching new resource: ' + e.request.url); */
          cache.put(e.request, response.clone());
          return response;
        });

      });
    })
  );
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keyList => {
          return Promise.all(
            keyList.map(key => {
              if(key !== cacheName) {
                return caches.delete(key);
              }
            })
          );
    })
  );
});
