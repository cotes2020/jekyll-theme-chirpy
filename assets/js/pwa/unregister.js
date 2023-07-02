---
layout: compress
permalink: '/unregister.js'
---

if ('serviceWorker' in navigator) {
    navigator.serviceWorker.getRegistrations().then((registrations) => {
        for (let reg of registrations) {
            reg.unregister();
        }
    });
}
