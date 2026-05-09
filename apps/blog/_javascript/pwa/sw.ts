/// <reference lib="webworker" />

const ctx = self as unknown as ServiceWorkerGlobalScope;
ctx.importScripts('./assets/js/data/swconf.js');

const purge = swconf.purge;
const interceptor = swconf.interceptor;

function verifyUrl(url: string): boolean {
  const requestUrl = new URL(url);
  const requestPath = requestUrl.pathname;

  if (!requestUrl.protocol.startsWith('http')) {
    return false;
  }

  if (!interceptor) {
    return true;
  }

  for (const prefix of interceptor.urlPrefixes) {
    if (requestUrl.href.startsWith(prefix)) {
      return false;
    }
  }

  for (const path of interceptor.paths) {
    if (requestPath.startsWith(path)) {
      return false;
    }
  }
  return true;
}

ctx.addEventListener('install', (event: ExtendableEvent) => {
  if (purge) {
    return;
  }

  const cacheName = swconf.cacheName;
  const resources = swconf.resources;
  if (!cacheName || !resources) {
    return;
  }

  event.waitUntil(
    caches.open(cacheName).then((cache) => {
      return cache.addAll(resources);
    })
  );
});

ctx.addEventListener('activate', (event: ExtendableEvent) => {
  event.waitUntil(
    caches.keys().then((keyList) => {
      return Promise.all(
        keyList.map((key) => {
          if (purge) {
            return caches.delete(key);
          }
          const cacheName = swconf.cacheName;
          if (cacheName && key !== cacheName) {
            return caches.delete(key);
          }
          return Promise.resolve();
        })
      );
    })
  );
});

ctx.addEventListener('message', (event: ExtendableMessageEvent) => {
  if (event.data === 'SKIP_WAITING') {
    void ctx.skipWaiting();
  }
});

ctx.addEventListener('fetch', (event: FetchEvent) => {
  if (event.request.headers.has('range')) {
    return;
  }

  event.respondWith(
    caches.match(event.request).then((response) => {
      if (response) {
        return response;
      }

      return fetch(event.request).then((response) => {
        const url = event.request.url;

        if (purge || event.request.method !== 'GET' || !verifyUrl(url)) {
          return response;
        }

        const cacheName = swconf.cacheName;
        if (!cacheName) {
          return response;
        }

        const responseToCache = response.clone();

        void caches.open(cacheName).then((cache) => {
          void cache.put(event.request, responseToCache);
        });
        return response;
      });
    })
  );
});
