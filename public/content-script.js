// ========================================================================
// DiFri Content Script — injected into every page by Rust (navigate_and_inject)
// Содержит: блокировка трекеров, очистка URL, антифингерпринтинг, панель управления
// ========================================================================
(function() {
  'use strict';

  // ------------------------------------------------------------------
  // 1. ПРАВИЛА БЛОКИРОВКИ (из shared/filter-rules.json)
  // ------------------------------------------------------------------
  // Правила фильтрации (могут быть обновлены через window.__difriRulesUpdate)
  let RULES = {
    "version": "2026.11.2",
    "name": "DiFri Privacy Balanced Shield List",
    "blockedDomains": [
      "doubleclick.net","googlesyndication.com","googleadservices.com",
      "google-analytics.com","googletagmanager.com","googletagservices.com",
      "adservice.google.com","pagead2.googlesyndication.com",
      "fundingchoicesmessages.google.com","2mdn.net","app-measurement.com",
      "mc.yandex.ru","an.yandex.ru","yabs.yandex.ru","adsfox.yandex.ru","adfox.ru",
      "adriver.ru","top-fwz1.mail.ru","my.rtmark.net","target.my.com",
      "connect.facebook.net","pixel.facebook.com","an.facebook.com",
      "analytics.tiktok.com","ads.tiktok.com","business-api.tiktok.com",
      "ads-twitter.com","analytics.twitter.com","static.ads-twitter.com","t.co/i/adsct",
      "snap.licdn.com","px.ads.linkedin.com","tr.snapchat.com","sc-static.net/scevent.min.js",
      "ct.pinterest.com","ads.pinterest.com",
      "events.redditmedia.com","alb.reddit.com",
      "criteo.com","criteo.net",
      "taboola.com","trc.taboola.com",
      "outbrain.com","widgets.outbrain.com",
      "adnxs.com","ib.adnxs.com",
      "rubiconproject.com","fastlane.rubiconproject.com",
      "pubmatic.com","ads.pubmatic.com",
      "openx.net","casalemedia.com","indexww.com",
      "adsrvr.org","thetradedesk.com",
      "smartadserver.com","teads.tv",
      "amazon-adsystem.com","aax.amazon-adsystem.com",
      "moatads.com","doubleverify.com","adsafeprotected.com",
      "serving-sys.com","flashtalking.com","adform.net","track.adform.net",
      "mediamath.com","mathtag.com","bidswitch.net","demdex.net","omtrdc.net",
      "everesttech.net","krxd.net","bluekai.com","exelator.com","rlcdn.com",
      "tapad.com","liadm.com","id5-sync.com","uidapi.com","lotame.com","crwdcntrl.net",
      "scorecardresearch.com","sb.scorecardresearch.com",
      "quantserve.com","quantcount.com",
      "chartbeat.com","static.chartbeat.com",
      "newrelic.com","bam.nr-data.net",
      "hotjar.com","static.hotjar.com","vars.hotjar.com",
      "clarity.ms","www.clarity.ms",
      "fullstory.com","rs.fullstory.com",
      "mouseflow.com","luckyorange.com","smartlook.com",
      "logrocket.io","cdn.logrocket.io",
      "mixpanel.com","api-js.mixpanel.com",
      "amplitude.com","api.amplitude.com",
      "segment.io","cdn.segment.com","api.segment.io",
      "heapanalytics.com","posthog.com",
      "intercom.io","widget.intercom.io",
      "hs-analytics.net","hsadspixel.net","pardot.com","marketo.net","munchkin.marketo.net",
      "braze.com","sdk.iad-01.braze.com",
      "appsflyer.com","adjust.com","branch.io","app.link",
      "fingerprintjs.com","fpnpmcdn.net","fpjs.io","api.fpjs.io","clientjs.org",
      "popads.net","popcash.net","propellerads.com","adcash.com","exoclick.com",
      "juicyads.com","trafficjunky.net","hilltopads.net","clickadu.com","adsterra.com"
    ],
    "blockedUrlPatterns": [
      "/pagead/","/gtag/js","/gtm.js","/analytics.js","/fbevents.js",
      "/metrika/tag.js","/metrika/watch.js","/ads/ga-audiences",
      "/pixel/track","/adserver/","/adframe.","/prebid","/fingerprint2","/openrtb2/"
    ],
    "trackingQueryParams": [
      "utm_source","utm_medium","utm_campaign","utm_term","utm_content","utm_id","utm_name",
      "fbclid","gclid","dclid","gbraid","wbraid","yclid","ysclid","msclkid","twclid","ttclid",
      "li_fat_id","mc_cid","mc_eid","_hsenc","_hsmi","igshid","vero_id","mkt_tok",
      "oly_enc_id","oly_anon_id","rb_clickid","s_kwcid"
    ],
    "domainCategories": {
      "doubleclick.net":"Рекламная сеть Google",
      "googlesyndication.com":"Рекламная сеть Google",
      "google-analytics.com":"Аналитика Google",
      "mc.yandex.ru":"Яндекс.Метрика",
      "connect.facebook.net":"Пиксель Meta",
      "pixel.facebook.com":"Пиксель Meta",
      "analytics.tiktok.com":"Трекер TikTok",
      "criteo.com":"Ретаргетинг",
      "hotjar.com":"Запись экрана (Session Replay)",
      "clarity.ms":"Запись экрана Microsoft",
      "fullstory.com":"Запись экрана",
      "fingerprintjs.com":"Фингерпринтинг",
      "mixpanel.com":"Продуктовая слежка",
      "segment.io":"Сборщик телеметрии",
      "scorecardresearch.com":"Кросс-сайт профиль"
    }
  };

  // ------------------------------------------------------------------
  // 2. АНТИФИНГЕРПРИНТИНГ
  // ------------------------------------------------------------------
  (function installAntiFingerprinting() {
    const FAKE_UA   = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36";
    const FAKE_PLATFORM = "Win32";
    const FAKE_LANGUAGE = "ru-RU";
    const FAKE_LANGUAGES = ["ru-RU","ru","en-US","en"];
    const FAKE_HARDWARE = 4;
    const FAKE_DEVICE_MEMORY = 8;
    const FAKE_SCREEN_W = 1920;
    const FAKE_SCREEN_H = 1080;

    const spoofs = {
      userAgent: FAKE_UA,
      platform: FAKE_PLATFORM,
      language: FAKE_LANGUAGE,
      languages: FAKE_LANGUAGES,
      hardwareConcurrency: FAKE_HARDWARE,
      deviceMemory: FAKE_DEVICE_MEMORY,
      doNotTrack: "0",
      cookieEnabled: true,
      onLine: true,
      vendor: "Google Inc.",
      product: "Gecko",
      productSub: "20030107",
      appVersion: "5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
      appName: "Netscape"
    };

    for (const [key, value] of Object.entries(spoofs)) {
      try {
        Object.defineProperty(navigator, key, { configurable: true, get: () => value });
      } catch (_) {}
    }

    // Замаскировать webdriver (защита от детекта автоматизации)
    try { Object.defineProperty(navigator, 'webdriver', { configurable: true, get: () => undefined }); } catch (_) {}

    // --- Canvas poisoning (возвращаем 1x1 прозрачный пиксель) ---
    const EMPTY_PNG = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=';
    try {
      const origToDataURL = HTMLCanvasElement.prototype.toDataURL;
      HTMLCanvasElement.prototype.toDataURL = function(...args) {
        if (args[0] === 'image/png' || args[0] === 'image/jpeg' || !args[0]) return EMPTY_PNG;
        return origToDataURL.apply(this, args);
      };
      const origToBlob = HTMLCanvasElement.prototype.toBlob;
      HTMLCanvasElement.prototype.toBlob = function(callback, ...args) {
        const byteString = atob(EMPTY_PNG.split(',')[1]);
        const ab = new ArrayBuffer(byteString.length);
        const ia = new Uint8Array(ab);
        for (let i = 0; i < byteString.length; i++) ia[i] = byteString.charCodeAt(i);
        setTimeout(() => callback(new Blob([ab], { type: 'image/png' })), 0);
      };
    } catch (_) {}

    // --- WebGL spoofing ---
    try {
      if (typeof WebGLRenderingContext !== 'undefined') {
        const origGetParameter = WebGLRenderingContext.prototype.getParameter;
        WebGLRenderingContext.prototype.getParameter = function(parameter) {
          const spoofed = {
            0x0B10: 'Google Inc.',   // VENDOR
            0x0B11: 'SwiftShader',   // RENDERER
            0x1F00: 'WebGL 1.0',     // VERSION
            0x8B4C: 'WebGL 1.0'      // SHADING_LANGUAGE_VERSION
          };
          if (spoofed[parameter] !== undefined) return spoofed[parameter];
          return origGetParameter.call(this, parameter);
        };
      }
      if (typeof WebGL2RenderingContext !== 'undefined') {
        const orig2 = WebGL2RenderingContext.prototype.getParameter;
        WebGL2RenderingContext.prototype.getParameter = function(parameter) {
          const spoofed = { 0x0B10: 'Google Inc.', 0x0B11: 'SwiftShader' };
          if (spoofed[parameter] !== undefined) return spoofed[parameter];
          return orig2.call(this, parameter);
        };
      }
    } catch (_) {}

    // --- AudioContext — заглушаем createAnalyser (частотный отпечаток) ---
    try {
      if (window.AudioContext || window.webkitAudioContext) {
        const AC = window.AudioContext || window.webkitAudioContext;
        const origCreateAnalyser = AC.prototype.createAnalyser;
        AC.prototype.createAnalyser = function() {
          const node = origCreateAnalyser.call(this);
          const origGetFloatFrequency = node.getFloatFrequencyData;
          node.getFloatFrequencyData = function(array) {
            for (let i = 0; i < array.length; i++) array[i] = -100;
          };
          const origGetByteFrequency = node.getByteFrequencyData;
          node.getByteFrequencyData = function(array) {
            for (let i = 0; i < array.length; i++) array[i] = 0;
          };
          return node;
        };
      }
    } catch (_) {}

    // --- Battery API блокировка ---
    try {
      if (navigator.getBattery) {
        navigator.getBattery = function() {
          return Promise.reject(new Error('Blocked by DiFri Privacy'));
        };
      }
    } catch (_) {}

    // --- Connection API ---
    try {
      Object.defineProperty(navigator, 'connection', { configurable: true, get: () => undefined });
    } catch (_) {}

    // --- Plugins / MimeTypes пустые ---
    try {
      Object.defineProperty(navigator, 'plugins', { configurable: true, get: () => [] });
      Object.defineProperty(navigator, 'mimeTypes', { configurable: true, get: () => [] });
    } catch (_) {}

    // --- Screen эмуляция ---
    try {
      const screenSpoofs = {
        width: FAKE_SCREEN_W, height: FAKE_SCREEN_H,
        availWidth: FAKE_SCREEN_W, availHeight: FAKE_SCREEN_H - 40,
        colorDepth: 24, pixelDepth: 24
      };
      for (const [key, val] of Object.entries(screenSpoofs)) {
        Object.defineProperty(screen, key, { configurable: true, get: () => val });
      }
    } catch (_) {}

    console.log('[DiFri] Anti-fingerprinting active');
  })();


  // ------------------------------------------------------------------
  // 2.1. DNS-over-HTTPS (DoH) конфигурация
  // ------------------------------------------------------------------
  const DOH_PROVIDERS = {
    cloudflare:  { name: 'Cloudflare 1.1.1.1',  url: 'https://cloudflare-dns.com/dns-query' },
    google:      { name: 'Google 8.8.8.8',      url: 'https://dns.google/resolve' },
    quad9:       { name: 'Quad9 9.9.9.9',       url: 'https://dns.quad9.net/dns-query' },
    adguard:     { name: 'AdGuard 94.140.14.14', url: 'https://dns.adguard.com/dns-query' }
  };

  // Читаем DoH настройки из инъекции Rust (window.__difriDoh) или из sessionStorage
  const injectedDoh = window.__difriDoh || {};
  let dohEnabled = injectedDoh.enabled === true || sessionStorage.getItem('difri:doh:enabled') === '1';
  let dohProvider = injectedDoh.provider || sessionStorage.getItem('difri:doh:provider') || 'cloudflare';

  async function resolveViaDoH(hostname) {
    if (!dohEnabled) return null;
    const provider = DOH_PROVIDERS[dohProvider];
    if (!provider) return null;
    try {
      const url = provider.url + '?name=' + encodeURIComponent(hostname) + '&type=A';
      const resp = await fetch(url, {
        headers: { 'accept': 'application/dns-json' },
        mode: 'cors'
      });
      if (!resp.ok) return null;
      const json = await resp.json();
      if (json.Answer && json.Answer.length > 0) {
        return json.Answer.filter(a => a.type === 1).map(a => a.data);
      }
      return [];
    } catch (_) {
      return null;
    }
  }

  function updateDohBadge() {
    const el = document.getElementById('difri-cs-doh-badge');
    if (!el) return;
    if (dohEnabled) {
      el.textContent = 'DoH: ' + DOH_PROVIDERS[dohProvider].name;
      el.style.cssText = 'background:#1a3a1a;color:#50fa7b;padding:4px 10px;border-radius:12px;border:1px solid #2a5a2a;font-size:11px;';
    } else {
      el.textContent = 'DoH: выключен';
      el.style.cssText = 'background:#3a1a1a;color:#ff5555;padding:4px 10px;border-radius:12px;border:1px solid #552222;font-size:11px;';
    }
  }

  // ------------------------------------------------------------------
  // 2.5. БЛОКИРОВКА WebRTC (защита от утечки реального IP)
  // ------------------------------------------------------------------
  (function installWebRTCBlocker() {
    const webrtcGlobals = [
      'RTCPeerConnection',
      'RTCSessionDescription',
      'RTCIceCandidate',
      'RTCRtpSender',
      'RTCRtpReceiver',
      'RTCRtpTransceiver',
      'RTCSctpTransport',
      'RTCDtlsTransport',
      'RTCIceTransport',
      'RTCDataChannel',
      'RTCIdentityProvider',
      'RTCIdentityProviderGlobalScope',
      'RTCPeerConnectionIceEvent'
    ];
    const webkitPrefix = 'webkit' in window ? 'webkit' : '';

    for (const name of webrtcGlobals) {
      try {
        Object.defineProperty(window, name, {
          configurable: true,
          get: () => undefined,
          set: () => {}
        });
      } catch (_) {}
    }

    // webkit-prefixed (Chrome/WebView2 на Windows)
    for (const prefix of ['webkitRTCPeerConnection', 'webkitRTCSessionDescription', 'webkitRTCIceCandidate']) {
      try {
        Object.defineProperty(window, prefix, {
          configurable: true,
          get: () => undefined,
          set: () => {}
        });
      } catch (_) {}
    }

    // navigator.mediaDevices.getUserMedia — блокируем доступ к камере/микрофону
    try {
      if (navigator.mediaDevices) {
        const origGUM = navigator.mediaDevices.getUserMedia;
        navigator.mediaDevices.getUserMedia = function(constraints) {
          return Promise.reject(new Error('Blocked by DiFri Privacy: WebRTC/media access denied'));
        };
        // enumerateDevices тоже возвращает пустой список
        const origEnum = navigator.mediaDevices.enumerateDevices;
        navigator.mediaDevices.enumerateDevices = function() {
          return Promise.resolve([]);
        };
      } else {
        Object.defineProperty(navigator, 'mediaDevices', {
          configurable: true,
          get: () => ({
            getUserMedia: () => Promise.reject(new Error('Blocked by DiFri Privacy')),
            enumerateDevices: () => Promise.resolve([])
          })
        });
      }
    } catch (_) {}

    console.log('[DiFri] WebRTC blocked — IP leak protection active');
  })();

  // ------------------------------------------------------------------
  // 2.6. АВТООТКАЗ РАЗРЕШЕНИЙ (камера, микрофон, геолокация, уведомления, буфер, автоплей)
  // ------------------------------------------------------------------
  (function installPermissionDenier() {
    const injected = window.__difriPermissions || {};
    const denied = injected.denied === true || sessionStorage.getItem('difri:permissions:denied') === '1';

    if (!denied) return;

    // --- navigator.permissions.query → всегда 'denied' ---
    try {
      const origQuery = navigator.permissions.query;
      navigator.permissions.query = function(desc) {
        return Promise.resolve({ state: 'denied', onchange: null });
      };
    } catch (_) {}

    // --- Notification.requestPermission → 'denied' ---
    try {
      if (window.Notification) {
        Notification.requestPermission = function() {
          return Promise.resolve('denied');
        };
        try { Notification.permission = 'denied'; } catch (_) {}
      }
    } catch (_) {}

    // --- navigator.clipboard — чтение/запись запрещены ---
    try {
      Object.defineProperty(navigator, 'clipboard', {
        configurable: true,
        get: () => ({
          readText:  () => Promise.reject(new Error('Blocked by DiFri Privacy: clipboard')),
          writeText: () => Promise.reject(new Error('Blocked by DiFri Privacy: clipboard')),
          read:      () => Promise.reject(new Error('Blocked by DiFri Privacy: clipboard')),
          write:     () => Promise.reject(new Error('Blocked by DiFri Privacy: clipboard'))
        })
      });
    } catch (_) {}

    // --- Geolocation — всегда ошибка ---
    try {
      if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition = function(success, error, opts) {
          if (error) error(new Error('Blocked by DiFri Privacy: geolocation'));
        };
        navigator.geolocation.watchPosition = function() { return 0; };
        navigator.geolocation.clearWatch = function() {};
      }
    } catch (_) {}

    // --- Autoplay — медиа не запускается без жеста пользователя ---
    try {
      const origPlay = HTMLMediaElement.prototype.play;
      HTMLMediaElement.prototype.play = function() {
        const promise = origPlay.call(this);
        if (promise && promise.catch) {
          return promise.catch(() => Promise.reject(new Error('Blocked by DiFri Privacy: autoplay')));
        }
        return Promise.reject(new Error('Blocked by DiFri Privacy: autoplay'));
      };
    } catch (_) {}

    console.log('[DiFri] Permission auto-denial active');
  })();

  // ------------------------------------------------------------------
  // 2.7. БЛОКИРОВКА СТОРОННИХ КУКИ (3rd-party cookies)
  // ------------------------------------------------------------------
  (function installThirdPartyCookieBlocker() {
    const injected = window.__difriCookies || {};
    const enabled = injected.blockThirdPartyCookies === true || sessionStorage.getItem('difri:cookies:thirdparty') === '1';

    if (!enabled) return;

    const currentHost = () => {
      try { return new URL(location.href).hostname.toLowerCase(); } catch { return ''; }
    };

    const isSameSiteCookie = (cookieDomain) => {
      const host = currentHost();
      if (!host) return false;
      const c = (cookieDomain || '').toLowerCase();
      const h = host;
      if (!c) return true; // куки без Domain — same-site
      return h === c || h.endsWith('.' + c);
    };

    let cookieStore = '';

    const cookieDescriptor = {
      configurable: true,
      get: function() {
        const host = currentHost();
        return cookieStore.split(';').map(c => c.trim()).filter(c => {
          // Если в строке куки есть Domain=, проверяем его
          const m = c.match(/Domain=([^;]+)/i);
          return isSameSiteCookie(m ? m[1] : '');
        }).join('; ');
      },
      set: function(value) {
        const parts = value.split(';');
        const cookiePart = parts[0].trim();
        const domainPart = parts.find(p => p.trim().toLowerCase().startsWith('domain='));
        const cookieDomain = domainPart ? domainPart.split('=')[1].trim() : '';

        if (!isSameSiteCookie(cookieDomain)) {
          console.warn('[DiFri] Blocked 3rd-party cookie:', cookiePart);
          return;
        }
        cookieStore = value;
      }
    };

    try {
      Object.defineProperty(document, 'cookie', cookieDescriptor);
    } catch (_) {}

    // Block Storage Access API
    try {
      if (document.requestStorageAccess) {
        document.requestStorageAccess = function() {
          return Promise.reject(new Error('Blocked by DiFri Privacy: 3rd-party storage access'));
        };
      }
    } catch (_) {}

    console.log('[DiFri] Third-party cookie blocker active');
  })();

  // ------------------------------------------------------------------
  // 2.8. HTTPS-ONLY (принудительный переход на HTTPS, блокировка HTTP)
  // ------------------------------------------------------------------
  (function installHttpsOnly() {
    const injected = window.__difriHttps || {};
    const enabled = injected.httpsOnly === true || sessionStorage.getItem('difri:https:only') === '1';

    if (!enabled) return;

    const currentHost = () => {
      try { return new URL(location.href).hostname; } catch { return ''; }
    };

    // Инъекция мета-тега upgrade-insecure-requests для браузера
    try {
      const meta = document.createElement('meta');
      meta.httpEquiv = 'Content-Security-Policy';
      meta.content = 'upgrade-insecure-requests';
      document.head.appendChild(meta);
    } catch (_) {}

    // Перехват fetch — http -> https
    const origFetch = window.fetch;
    window.fetch = function(input, init) {
      let url = typeof input === 'string' ? input : (input && input instanceof Request ? input.url : String(input));
      try {
        const u = new URL(url);
        if (u.protocol === 'http:') {
          u.protocol = 'https:';
          url = u.toString();
          console.warn('[DiFri] HTTPS-only: upgraded', url);
        }
      } catch (_) {}
      if (typeof input === 'string') return origFetch(url, init);
      if (input instanceof Request) {
        init = init || {};
        init.method = input.method;
        init.headers = input.headers;
        init.body = input.body;
        return origFetch(new Request(url, init));
      }
      return origFetch(input, init);
    };

    // Перехват XMLHttpRequest — http -> https
    const origOpen = XMLHttpRequest.prototype.open;
    XMLHttpRequest.prototype.open = function(method, url) {
      try {
        const u = new URL(url);
        if (u.protocol === 'http:') {
          u.protocol = 'https:';
          url = u.toString();
        }
      } catch (_) {}
      this._difriUrl = url;
      return origOpen.call(this, method, url);
    };

    // Блокировка загрузки ресурсов (img, script, link, iframe) через http
    try {
      const origAppendChild = Node.prototype.appendChild;
      Node.prototype.appendChild = function(child) {
        if (child.tagName) {
          const tag = child.tagName.toLowerCase();
          if (tag === 'img' || tag === 'script' || tag === 'link' || tag === 'iframe') {
            const src = child.src || child.href;
            if (src) {
              try {
                const u = new URL(src);
                if (u.protocol === 'http:') {
                  u.protocol = 'https:';
                  if (tag === 'link') child.href = u.toString();
                  else child.src = u.toString();
                  console.warn('[DiFri] HTTPS-only: upgraded resource', u.toString());
                }
              } catch (_) {}
            }
          }
        }
        return origAppendChild.call(this, child);
      };
    } catch (_) {}

    // Перехват кликов по ссылкам http -> https
    try {
      document.addEventListener('click', (e) => {
        let el = e.target;
        while (el && el !== document.body) {
          if (el.tagName === 'A' && el.href) {
            try {
              const u = new URL(el.href);
              if (u.protocol === 'http:') {
                u.protocol = 'https:';
                el.href = u.toString();
              }
            } catch (_) {}
            break;
          }
          el = el.parentElement;
        }
      }, true);
    } catch (_) {}

    console.log('[DiFri] HTTPS-only active');
  })();

  // ------------------------------------------------------------------
  // 2.9. КОСМЕТИЧЕСКАЯ БЛОКИРОВКА (скрытие рекламных элементов на странице)
  // ------------------------------------------------------------------
  (function installCosmeticBlocker() {
    const injected = window.__difriCosmetic || {};
    const enabled = injected.cosmeticBlocking === true || sessionStorage.getItem('difri:cosmetic:enabled') === '1';

    if (!enabled) return;

    const SELECTORS = (RULES.cosmeticSelectors || []).filter(Boolean);
    const CATEGORIES = (RULES.cosmeticCategories || {});

    let hiddenCount = 0;

    function hideElement(el) {
      if (el.__difriHidden) return;
      el.__difriHidden = true;
      el.style.setProperty('display', 'none', 'important');
      el.style.setProperty('visibility', 'hidden', 'important');
      hiddenCount++;
      // Небольшая задержка перед удалением из DOM, чтобы не сломать layout скриптов сайта
      setTimeout(() => {
        try { el.remove(); } catch (_) {}
      }, 100);
    }

    function applyToDocument(doc) {
      SELECTORS.forEach(selector => {
        try {
          const elements = doc.querySelectorAll(selector);
          elements.forEach(hideElement);
        } catch (_) {}
      });
    }

    // Применяем к текущему документу
    applyToDocument(document);

    // Observer для динамически добавленных элементов (реклама часто подгружается через JS)
    try {
      const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
          mutation.addedNodes.forEach((node) => {
            if (node.nodeType !== 1) return; // только элементы
            // Проверяем сам элемент
            SELECTORS.forEach(selector => {
              try {
                if (node.matches && node.matches(selector)) hideElement(node);
                // Проверяем дочерние элементы
                const matches = node.querySelectorAll ? node.querySelectorAll(selector) : [];
                matches.forEach(hideElement);
              } catch (_) {}
            });
          });
        });
      });
      observer.observe(document.documentElement, { childList: true, subtree: true });
    } catch (_) {}

    console.log('[DiFri] Cosmetic blocking active. Selectors:', SELECTORS.length);
  })();

  // ------------------------------------------------------------------
  // 3. БЛОКИРОВКА СЕТЕВЫХ ЗАПРОСОВ (fetch / XHR)
  // ------------------------------------------------------------------
  const DOMAINS = new Set((RULES.blockedDomains || []).map(d => d.toLowerCase()));
  const PATTERNS = (RULES.blockedUrlPatterns || []).map(p => p.toLowerCase());
  const STRIP_PARAMS = new Set((RULES.trackingQueryParams || []));

  let blockedCount = Number(sessionStorage.getItem('difri:cs:blocked') || '0');
  const blockStats = JSON.parse(sessionStorage.getItem('difri:cs:stats') || '{}');
  const blockHistory = JSON.parse(sessionStorage.getItem('difri:cs:history') || '[]');

  function hostnameOf(url) {
    try { return new URL(url).hostname.toLowerCase(); } catch { return ''; }
  }

  function isBlocked(url) {
    try {
      const u = new URL(url);
      const host = u.hostname.toLowerCase();
      const full = (host + u.pathname + u.search).toLowerCase();
      for (const d of DOMAINS) {
        if (host === d || host.endsWith('.' + d)) return { blocked: true, reason: d, type: 'domain' };
      }
      for (const p of PATTERNS) {
        if (full.includes(p)) return { blocked: true, reason: p, type: 'pattern' };
      }
      return { blocked: false };
    } catch { return { blocked: false }; }
  }

  function stripTracking(url) {
    try {
      const u = new URL(url);
      for (const p of STRIP_PARAMS) u.searchParams.delete(p);
      return u.toString();
    } catch { return url; }
  }

  function incrementBlocked(reason, type, url) {
    blockedCount++;
    sessionStorage.setItem('difri:cs:blocked', String(blockedCount));
    updateToolbarBadge();

    const host = hostnameOf(url) || reason;
    const category = RULES.domainCategories?.[reason] || (type === 'pattern' ? 'Паттерн URL' : 'Домен');

    // Считаем по доменам
    if (!blockStats[host]) blockStats[host] = { count: 0, category: category, type: type };
    blockStats[host].count++;

    // История последних 50 блокировок
    blockHistory.unshift({
      time: new Date().toISOString(),
      domain: host,
      reason: reason,
      type: type,
      category: category,
      url: url
    });
    if (blockHistory.length > 50) blockHistory.pop();

    sessionStorage.setItem('difri:cs:stats', JSON.stringify(blockStats));
    sessionStorage.setItem('difri:cs:history', JSON.stringify(blockHistory));
  }

  // --- Patch fetch ---
  const origFetch = window.fetch;
  window.fetch = function(input, init) {
    let url = typeof input === 'string' ? input : (input && input instanceof Request ? input.url : String(input));
    const cleaned = stripTracking(url);
    const check = isBlocked(cleaned);
    if (check.blocked) {
      incrementBlocked(check.reason, check.type, cleaned);
      console.warn('[DiFri] Blocked fetch:', cleaned, '(' + check.reason + ')');
      return Promise.reject(new Error('Blocked by DiFri Privacy: ' + check.reason));
    }
    if (cleaned !== url && input instanceof Request) {
      init = init || {};
      init.method = input.method;
      init.headers = input.headers;
      init.body = input.body;
      return origFetch(new Request(cleaned, init));
    }
    if (cleaned !== url && typeof input === 'string') return origFetch(cleaned, init);
    return origFetch(input, init);
  };

  // --- Patch XMLHttpRequest ---
  const origOpen = XMLHttpRequest.prototype.open;
  const origSend = XMLHttpRequest.prototype.send;
  XMLHttpRequest.prototype.open = function(method, url) {
    this._difriUrl = url;
    return origOpen.apply(this, arguments);
  };
  XMLHttpRequest.prototype.send = function(body) {
    if (this._difriUrl) {
      const cleaned = stripTracking(this._difriUrl);
      const check = isBlocked(cleaned);
      if (check.blocked) {
        incrementBlocked(check.reason, check.type, cleaned);
        console.warn('[DiFri] Blocked XHR:', cleaned, '(' + check.reason + ')');
        this._difriBlocked = true;
        return;
      }
      this._difriUrl = cleaned;
    }
    return origSend.apply(this, arguments);
  };
  const origAddEventListener = EventTarget.prototype.addEventListener;
  EventTarget.prototype.addEventListener = function(type, listener, options) {
    if (type === 'readystatechange' && this instanceof XMLHttpRequest) {
      const wrapped = function(ev) {
        if (this._difriBlocked && this.readyState >= 2) {
          try { this.abort(); } catch (_) {}
          this._difriBlocked = false;
          return;
        }
        return listener.call(this, ev);
      };
      return origAddEventListener.call(this, type, wrapped, options);
    }
    return origAddEventListener.call(this, type, listener, options);
  };

  // ------------------------------------------------------------------
  // 4. ПАНЕЛЬ УПРАВЛЕНИЯ (на внешней странице)
  // ------------------------------------------------------------------
  function toastCS(msg) {
    // Показываем уведомление на странице (в консоли и через badge)
    console.log('[DiFri] ' + msg);
    const badge = document.getElementById('difri-cs-badge');
    if (badge) {
      badge.textContent = 'Фильтры обновлены';
      setTimeout(() => { badge.textContent = 'Заблокировано: ' + blockedCount; }, 2000);
    }
  }

  function updateToolbarBadge() {
    const el = document.getElementById('difri-cs-badge');
    if (el) el.textContent = 'Заблокировано: ' + blockedCount;
  }

  function injectToolbar() {
    if (document.getElementById('difri-cs-toolbar')) return;
    const tb = document.createElement('div');
    tb.id = 'difri-cs-toolbar';
    tb.style.cssText = [
      'position:fixed;top:0;left:0;right:0;height:44px;z-index:2147483647;',
      'background:#1a1a1a;border-bottom:1px solid #333;color:#eaeaea;',
      'font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;',
      'display:flex;align-items:center;padding:0 10px;gap:8px;',
      'box-shadow:0 2px 10px rgba(0,0,0,0.4);font-size:13px;'
    ].join('');

    const btn = (label, onClick, title) => {
      const b = document.createElement('button');
      b.textContent = label;
      b.title = title;
      b.style.cssText = 'background:#262626;border:1px solid #444;color:#eaeaea;padding:6px 12px;border-radius:6px;cursor:pointer;font-size:12px;';
      b.onmouseover = () => { b.style.borderColor = '#bd93f9;'; };
      b.onmouseout = () => { b.style.borderColor = '#444;'; };
      b.onclick = onClick;
      return b;
    };

    const urlDisplay = document.createElement('div');
    urlDisplay.style.cssText = 'flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding:0 8px;color:#a0a0a0;';
    try { urlDisplay.textContent = new URL(location.href).hostname; } catch { urlDisplay.textContent = location.href; }

    tb.appendChild(btn('←', () => history.back(), 'Назад'));
    tb.appendChild(btn('→', () => history.forward(), 'Вперёд'));
    tb.appendChild(urlDisplay);
    const appOrigin = window.__difriAppOrigin || location.origin;
    tb.appendChild(btn('Новая вкладка', () => { location.href = appOrigin + '/app.html'; }, 'Вернуться в DiFri'));
    tb.appendChild(btn('Очистить страницу', () => {
      localStorage.clear(); sessionStorage.clear(); blockedCount = 0;
      sessionStorage.setItem('difri:cs:blocked', '0');
      updateToolbarBadge(); location.reload();
    }, 'Очистить хранилище и перезагрузить'));
    tb.appendChild(document.createElement('div')).style.flex = '1';
    const badge = document.createElement('span');
    badge.id = 'difri-cs-badge';
    badge.textContent = 'Заблокировано: ' + blockedCount;
    badge.style.cssText = 'background:#3a1a1a;color:#ff5555;padding:4px 10px;border-radius:12px;border:1px solid #552222;font-size:11px;';
    tb.appendChild(badge);

    const dohBadge = document.createElement('span');
    dohBadge.id = 'difri-cs-doh-badge';
    dohBadge.textContent = 'DoH: initializing...';
    dohBadge.style.cssText = 'background:#2a2a2a;color:#a0a0a0;padding:4px 10px;border-radius:12px;border:1px solid #444;font-size:11px;margin-left:8px;';
    tb.appendChild(dohBadge);
    updateDohBadge();

    document.documentElement.insertBefore(tb, document.documentElement.firstChild);
    updateToolbarBadge();
    updateDohBadge();
  }

  // DoH переключатель доступен через window.__difriCS
  window.__difriCS.setDoh = function(enabled, provider) {
    dohEnabled = enabled;
    if (provider) dohProvider = provider;
    sessionStorage.setItem('difri:doh:enabled', dohEnabled ? '1' : '0');
    sessionStorage.setItem('difri:doh:provider', dohProvider);
    updateDohBadge();
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', injectToolbar);
  } else {
    injectToolbar();
  }

  window.__difriCS = {
    blockedCount: () => blockedCount,
    rules: RULES,
    isBlocked: isBlocked,
    stripTracking: stripTracking
  };

  console.log('[DiFri Content Script] Active. Rules:', DOMAINS.size, 'domains | Anti-fingerprinting: ON');
})();
