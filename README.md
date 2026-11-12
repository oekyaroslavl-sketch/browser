# DiFri Privacy — лёгкий приватный браузер

DiFri Privacy — ответвление (fork) идеи [DiFri](https://github.com/Abhijit-71/DiFri): минималистичный, быстрый и приватный браузер на **Tauri v2** (одна кодовая база для **Windows/Linux/macOS** на рабочем столе и **Android** на телефоне).

## Стек

| Часть | Технология |
|-------|------------|
| Оболочка (desktop) | Tauri v2 + системный WebView (WebView2 / WebKitGTK / WebView) |
| Оболочка (Android) | Та же сборка Tauri, системный WebView Android |
| Интерфейс | Vanilla JS / HTML / CSS (без фреймворков), vite |
| Ядро | Rust (Tauri), команды IPC |

Почему не PyQt6: QWebEngineView тянет Chromium (~100–150 МБ бинарника) и не работает на смартфонах. Tauri использует системный WebView — сборка в 5–15 раз легче и работает на Android из коробки.

## Что реализовано

- **Вкладки** — переключение между страницами в одном окне (только одна страница загружена в каждый момент →'économie оперативной памяти).
- **Адресная строка** — навигация по URL или поиск DuckDuckGo.
- **Кнопки назад/вперёд/перезагрузка**, новая вкладка.
- **Блокировщик слежки** — встроенный список из 200+ рекламных и аналитических доменов (Google, Яндекс, Meta, TikTok, Criteo, Hotjar, Microsoft Clarity, finгерпринтинг-скрипты и др.). При попытке перехода открывается интерстициал-заглушка вместо сайта.
- **Очистка URL** — автоматическое удаление утилитных маркеров слежки (`utm_*`, `fbclid`, `gclid`, `yclid` и др.) из адресов.
- **Режим «Сжечь сессию»** — одна команда очищает куки, локальное хранилище, кэш и IndexedDB текущего вебвью (`core:webview:allow-clear-all-browsing-data`).
- **Блокировка доступа к сенсорам по умолчанию** — камера, микрофон, геолокация, уведомления запрашивают разрешение через систему WebView (по умолчанию запрещено).
- **Сборка под Windows** — `.msi` / `.nsis` установщики через GitHub Actions.
- **Сборка под Android** — APK через GitHub Actions (нет необходимости устанавливать Android SDK локально).

## Структура репозитория

```
difri-privacy/
├── public/               # интерфейс браузера (загружается в вебвью)
│   ├── app.html          # основная оболочка (панель, вкладки, вебвью)
│   ├── index.html        # страница новой вкладки
│   ├── blocked.html      # заглушка при блокировке ресурса
│   ├── filter-rules.json # правила блокировки (домены, паттерны, utm-метки)
│   └── svg/              # иконки
├── shared/filter-rules.json   # единый источник правил (копируется в public/)
├── src-tauri/
│   ├── Cargo.toml        # зависимости Rust
│   ├── tauri.conf.json   # конфигурация приложения (CSP, user agent, иконки)
│   ├── src/
│   │   ├── lib.rs        # ядро: команды clear_browsing_data, app_version
│   │   └── main.rs       # desktop entry point
│   └── icons/            # иконки приложения
├── .github/workflows/
│   ├── windows.yml       # сборка .msi/.nsis под Windows
│   └── android.yml       # сборка APK под Android
└── README.md
```

## Как собрать и запустить

### Локально (desktop)

```bash
# 1. Установите зависимости
npm ci

# 2. Режим разработки (открывает окно браузера с hot reload)
npm run tauri:dev

# 3. Сборка установщиков
npm run tauri:build
# Готовые пакеты: src-tauri/target/release/bundle/
```

### Android (локально)

Понадобится Rust с целями Android, JDK 17 и Android SDK/NDK:

```bash
rustup target add aarch64-linux-android armv7-linux-androideabi x86_64-linux-android
npm run tauri:android:init
npm run tauri:android:build
# APK: src-tauri/gen/android/app/build/outputs/apk/
```

### В облаке (GitHub Actions)

Запушьте код в ветку `master` или вручную запустите workflow:

- **Build Windows** — artefakты: `difri-privacy-windows` (.msi/.nsis).
- **Build Android** — artefakt: `difri-privacy-android-apk` (.apk, unsigned debug).

## Конфигурация приватности

`src-tauri/tauri.conf.json` — здесь задаётся пользовательский user-agent, политика безопасности контента (CSP) и список включённых плагинов. Список блокировки правится в `public/filter-rules.json` (и `shared/filter-rules.json` — источник).

## Ограничения текущей версии

- Это MVP-скелет: нет синхронизации закладок/истории между устройствами, нет расширений и встроенного VPN/Tor.
- Вкладки реализованы как последовательная загрузка в одном вебвью (экономит память, но страница перезагружается при переключении).
- Дальнейшее: изоляция вкладок через дочерние вебвью (Tauri multi-webview), интеграция с DNS-over-HTTPS, режим «Тор» и автообновление списка блокировки.

## Лицензия

MIT (см. `LICENSE.txt`). Иконки и часть стилей взяты из проекта DiFri (MIT).

## roadmap

- [x] Скелет приложения (Tauri v2, HTML/JS оболочка)
- [x] Вкладки, адресная строка, навигация
- [x] Блокировка рекламы/тре커ов (200+ доменов)
- [x] Очистка сессии («Сжечь сессию»)
- [x] CI-сборка Windows (.msi/.nsis) и Android (APK)
- [ ] Изоляция вкладок (multi-webview)
- [ ] Блокировка скриптов_tracker на уровне контент-скрипта
- [ ] VPN/Tor режим и DoH
