const CACHE_NAME = 'ngo-fraud-pwa-cache-v1';
const urlsToCache = [
    '/', // The main PWA shell (index.html)
    '/manifest.json',
    '/static/icons/icon-192x192.png',
    '/static/icons/icon-512x512.png',
    // Add other static assets if your PWA shell has more CSS/JS files
];

self.addEventListener('install', (event) => {
    console.log('Service Worker: Installing...');
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then((cache) => {
                console.log('Service Worker: Caching app shell');
                return cache.addAll(urlsToCache);
            })
            .catch(error => {
                console.error('Service Worker: Cache addAll failed', error);
            })
    );
});

self.addEventListener('fetch', (event) => {
    // Intercept requests for the PWA shell assets
    event.respondWith(
        caches.match(event.request)
            .then((response) => {
                if (response) {
                    console.log('Service Worker: Fetching from cache:', event.request.url);
                    return response;
                }
                console.log('Service Worker: Fetching from network:', event.request.url);
                return fetch(event.request);
            })
            .catch(error => {
                console.error('Service Worker: Fetch failed', error);
                // You could return an offline page here if needed
            })
    );
});

self.addEventListener('activate', (event) => {
    console.log('Service Worker: Activating...');
    event.waitUntil(
        caches.keys().then((cacheNames) => {
            return Promise.all(
                cacheNames.map((cacheName) => {
                    if (cacheName !== CACHE_NAME) {
                        console.log('Service Worker: Deleting old cache:', cacheName);
                        return caches.delete(cacheName);
                    }
                })
            );
        })
    );
});