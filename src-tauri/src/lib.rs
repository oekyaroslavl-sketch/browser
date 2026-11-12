use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};
use tauri::{command, generate_context, generate_handler, State, Webview};

/// Снимок статистики блокировок последней загруженной страницы (JSON).
static PAGE_STATS: Mutex<String> = Mutex::new(String::new());

/// Параметры инъекции контент-скрипта, сохранённые последним вызовом `navigate_tab`.
///
/// Зачем: `webview.navigate(url)` заменяет документ, и скрипт, введённый через `eval`
/// сразу после навигации, теряется вместе со старым документом. Поэтому скрипт
/// вводится повторно в обработчике `PageLoad`, когда новый документ уже создан.
static INJECTION: Mutex<Option<InjectionConfig>> = Mutex::new(None);

/// Номер текущей навигации. Растёт при каждом `navigate_tab`, чтобы фоновые
/// задачи прошлых переходов не вводили устаревшие настройки в новый документ.
static NAV_GENERATION: Mutex<u64> = Mutex::new(0);

#[derive(Clone)]
struct InjectionConfig {
    app_origin: String,
    label: String,
    doh_enabled: bool,
    doh_provider: String,
    permissions_denied: bool,
    block_third_party_cookies: bool,
    https_only: bool,
    rules_json: String,
    cosmetic_blocked: bool,
}

/// Нормализует ввод пользователя в абсолютный URL.
fn normalize_url(target: &str) -> Result<String, String> {
    let trimmed = target.trim();
    if trimmed.is_empty() {
        return Err("empty URL".to_string());
    }
    if trimmed.starts_with("about:") || trimmed.starts_with("data:") {
        return Ok(trimmed.to_string());
    }
    let mut href = trimmed.to_string();
    if !href.contains("://") {
        let looks_like_host = href.contains('.') && !href.contains(' ');
        href = if looks_like_host {
            format!("https://{}", href)
        } else {
            format!(
                "https://html.duckduckgo.com/html/?q={}",
                percent_encode(&href)
            )
        };
    }
    match href.parse::<tauri::Url>() {
        Ok(u) => {
            let scheme = u.scheme();
            if scheme != "http" && scheme != "https" {
                return Err(format!("unsupported scheme: {}", scheme));
            }
            Ok(href)
        }
        Err(e) => Err(format!("invalid URL: {}", e)),
    }
}

/// Percent-кодирование (поисковые запросы и адреса для error.html).
fn percent_encode(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for b in input.as_bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(*b as char)
            }
            _ => out.push_str(&format!("%{:02X}", b)),
        }
    }
    out
}

/// Собирает префикс с настройками приватности + сам контент-скрипт.
fn build_content_script(cfg: &InjectionConfig) -> String {
    let j = |v: &str| serde_json::to_string(v).unwrap_or_else(|_| "\"\"".to_string());
    let prefix = format!(
        "window.__difriAppOrigin = {};\n\
         window.__difriDoh = {{ enabled: {}, provider: {} }};\n\
         window.__difriPermissions = {{ denied: {} }};\n\
         window.__difriCookies = {{ blockThirdPartyCookies: {} }};\n\
         window.__difriHttps = {{ httpsOnly: {} }};\n\
         window.__difriTabLabel = {};\n\
         window.__difriRulesUpdate = {{ json: {} }};\n\
         window.__difriCosmetic = {{ cosmeticBlocking: {} }};\n\
         window.__difriRules = {};\n",
        j(&cfg.app_origin),
        cfg.doh_enabled,
        j(&cfg.doh_provider),
        cfg.permissions_denied,
        cfg.block_third_party_cookies,
        cfg.https_only,
        j(&cfg.label),
        cfg.rules_json,
        cfg.cosmetic_blocked,
        cfg.rules_json,
    );
    // Guard делает инъекцию идемпотентной: повторный eval в том же документе
    // ничего не делает, а после навигации window новый и скрипт вводится снова.
    // Благодаря этому не нужен on_page_load (в tauri 2.12 его нет ни у Webview,
    // ни у WebviewWindow) — достаточно периодических попыток eval.
    format!(
        "if (!window.__difriInjected) {{ window.__difriInjected = true;\n{}\n{}\n}}",
        prefix,
        include_str!("../../public/content-script.js")
    )
}

/// Забирает снимок статистики блокировок из sessionStorage страницы.
///
/// `eval_with_callback` здесь безопасен: результат нужен синхронно, а вызов
/// делается только по явному запросу (отчёт о приватности), а не при навигации —
/// раньше ожидание с `recv_timeout(5с)` висело на каждом переходе и вешало UI.
async fn snapshot_page_stats(webview: Webview) -> Option<String> {
    let snapshot = r#"(function(){ try { return JSON.stringify({ blockedCount: parseInt(sessionStorage.getItem('difri:cs:blocked')||'0',10), stats: JSON.parse(sessionStorage.getItem('difri:cs:stats')||'{}') }); } catch(e) { return '{}'; } })()"#;
    let (tx, rx) = std::sync::mpsc::channel::<String>();
    webview
        .eval_with_callback(snapshot, move |res| {
            let _ = tx.send(res);
        })
        .ok()?;
    let data = tokio::task::spawn_blocking(move || {
        rx.recv_timeout(std::time::Duration::from_secs(3)).ok()
    })
    .await
    .ok()??;
    if data.trim().is_empty() {
        None
    } else {
        Some(data)
    }
}

#[derive(Clone, Serialize, Deserialize)]
enum DownloadStatus {
    Started,
    Progress,
    Completed,
    Failed,
    Cancelled,
}

/// Информация о загрузке.
#[derive(Clone, Serialize, Deserialize)]
struct DownloadInfo {
    id: String,
    url: String,
    file_name: String,
    path: Option<String>,
    total_bytes: Option<u64>,
    received_bytes: u64,
    status: DownloadStatus,
    error: Option<String>,
}

/// Внутреннее состояние менеджера загрузок (оборачивается в Arc для передачи в таски).
struct DownloadManagerInner {
    downloads: RwLock<HashMap<String, DownloadInfo>>,
    cancel_tx: RwLock<HashMap<String, tokio::sync::mpsc::Sender<()>>>,
}

/// Менеджер загрузок: хранит состояние и каналы отмены.
struct DownloadManager {
    inner: Arc<DownloadManagerInner>,
}

impl Clone for DownloadManager {
    fn clone(&self) -> Self {
        Self { inner: self.inner.clone() }
    }
}

impl Default for DownloadManager {
    fn default() -> Self {
        Self {
            inner: Arc::new(DownloadManagerInner {
                downloads: RwLock::new(HashMap::new()),
                cancel_tx: RwLock::new(HashMap::new()),
            }),
        }
    }
}

impl DownloadManager {
    fn downloads(&self) -> &RwLock<HashMap<String, DownloadInfo>> {
        &self.inner.downloads
    }
    fn cancel_tx(&self) -> &RwLock<HashMap<String, tokio::sync::mpsc::Sender<()>>> {
        &self.inner.cancel_tx
    }
}

/// Начать скачивание файла в фоне. Возвращает id загрузки.
#[command]
async fn download_start(
    url: String,
    state: State<'_, DownloadManager>,
) -> Result<String, String> {
    use std::io::Write;

    let id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .to_string();

    // Определяем имя файла из URL
    let file_name = url
        .rsplit('/')
        .next()
        .unwrap_or("download")
        .split('?')
        .next()
        .unwrap_or("download")
        .to_string();
    if file_name.is_empty() {
        return Err("cannot determine file name from URL".to_string());
    }

    // Папка Downloads пользователя
    let downloads_dir = dirs::download_dir()
        .ok_or_else(|| "cannot determine downloads folder".to_string())?;
    let file_path = downloads_dir.join(&file_name);

    // Создаём канал отмены
    let (cancel_tx, mut cancel_rx) = tokio::sync::mpsc::channel::<()>(1);

    // Регистрируем начальное состояние
    {
        let mut downloads = state.downloads().write().unwrap();
        downloads.insert(
            id.clone(),
            DownloadInfo {
                id: id.clone(),
                url: url.clone(),
                file_name: file_name.clone(),
                path: None,
                total_bytes: None,
                received_bytes: 0,
                status: DownloadStatus::Started,
                error: None,
            },
        );
        state
            .cancel_tx()
            .write()
            .unwrap()
            .insert(id.clone(), cancel_tx);
    }

    // Клонируем менеджер для фонового таска (Arc — дешёвое клонирование)
    let mgr = DownloadManager { inner: state.inner.clone() };
    let id_out = id.clone();

    tauri::async_runtime::spawn(async move {
        let client = reqwest::Client::new();
        let total: Option<u64>;
        let mut received: u64 = 0;
        let mut failed = false;

        let set_status = |mgr: &DownloadManager,
                          id: &str,
                          status: DownloadStatus,
                          total: Option<u64>,
                          received: Option<u64>,
                          error: Option<String>| {
            let mut downloads = mgr.downloads().write().unwrap();
            if let Some(info) = downloads.get_mut(id) {
                info.status = status;
                if let Some(t) = total {
                    info.total_bytes = Some(t);
                }
                if let Some(r) = received {
                    info.received_bytes = r;
                }
                if error.is_some() {
                    info.error = error;
                }
            }
        };

        match client.get(&url).send().await {
            Ok(mut resp) => {
                total = resp.content_length();
                let mut file = match std::fs::File::create(&file_path) {
                    Ok(f) => f,
                    Err(e) => {
                        set_status(
                            &mgr,
                            &id,
                            DownloadStatus::Failed,
                            total,
                            Some(received),
                            Some(format!("file create: {}", e)),
                        );
                        mgr.cancel_tx().write().unwrap().remove(&id);
                        return;
                    }
                };

                loop {
                    // Проверяем отмену
                    if cancel_rx.try_recv().is_ok() {
                        let _ = std::fs::remove_file(&file_path);
                        set_status(
                            &mgr,
                            &id,
                            DownloadStatus::Cancelled,
                            total,
                            Some(received),
                            None,
                        );
                        break;
                    }

                    match resp.chunk().await {
                        Ok(Some(chunk)) => {
                            if let Err(e) = file.write_all(&chunk) {
                                set_status(
                                    &mgr,
                                    &id,
                                    DownloadStatus::Failed,
                                    total,
                                    Some(received),
                                    Some(format!("write: {}", e)),
                                );
                                failed = true;
                                break;
                            }
                            received += chunk.len() as u64;
                            set_status(
                                &mgr,
                                &id,
                                DownloadStatus::Progress,
                                total,
                                Some(received),
                                None,
                            );
                        }
                        Ok(None) => break,
                        Err(e) => {
                            set_status(
                                &mgr,
                                &id,
                                DownloadStatus::Failed,
                                total,
                                Some(received),
                                Some(format!("network: {}", e)),
                            );
                            failed = true;
                            break;
                        }
                    }
                }

                if !failed {
                    // Сохраняем канонический путь
                    if let Ok(canonical) = std::fs::canonicalize(&file_path) {
                        let mut downloads = mgr.downloads().write().unwrap();
                        if let Some(info) = downloads.get_mut(&id) {
                            info.path = Some(canonical.to_string_lossy().to_string());
                        }
                    }
                    set_status(
                        &mgr,
                        &id,
                        DownloadStatus::Completed,
                        total,
                        Some(received),
                        None,
                    );
                }

                // Удаляем канал отмены
                mgr.cancel_tx().write().unwrap().remove(&id);
            }
            Err(e) => {
                set_status(
                    &mgr,
                    &id,
                    DownloadStatus::Failed,
                    None,
                    Some(0),
                    Some(format!("request: {}", e)),
                );
                mgr.cancel_tx().write().unwrap().remove(&id);
            }
        }
    });

    Ok(id_out)
}

/// Отменить загрузку по id.
#[command]
fn download_cancel(id: String, state: State<'_, DownloadManager>) -> Result<(), String> {
    if let Some(tx) = state.cancel_tx().read().unwrap().get(&id).cloned() {
        let _ = tx.send(());
    }
    Ok(())
}

/// Открыть загруженный файл в системном файловом менеджере.
#[command]
fn download_open(id: String, state: State<'_, DownloadManager>) -> Result<(), String> {
    let path = {
        let downloads = state.downloads().read().unwrap();
        let info = downloads
            .get(&id)
            .ok_or_else(|| "download not found".to_string())?;
        info.path
            .as_deref()
            .ok_or_else(|| "file path not available".to_string())?
            .to_string()
    };

    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("explorer")
            .arg(format!("/select,{}", path))
            .spawn()
            .map_err(|e| e.to_string())?;
    }
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open").arg(&path).spawn().map_err(|e| e.to_string())?;
    }
    #[cfg(target_os = "linux")]
    {
        std::process::Command::new("xdg-open").arg(&path).spawn().map_err(|e| e.to_string())?;
    }
    let _ = path;
    Ok(())
}

/// Получить список всех загрузок.
#[command]
fn get_downloads(state: State<'_, DownloadManager>) -> Result<Vec<DownloadInfo>, String> {
    let downloads = state.downloads().read().unwrap();
    Ok(downloads.values().cloned().collect())
}

/// Очистить все данные просмотра текущего вебвью:
/// куки, локальное хранилище, IndexedDB, кэш и сессионные данные.
/// Вызывается из меню «Сжечь сессию».
#[command]
fn clear_browsing_data(webview: tauri::Webview) -> Result<(), String> {
    webview.clear_all_browsing_data().map_err(|e| e.to_string())
}

/// Версия приложения (из Cargo.toml).
#[command]
fn app_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Результат резолвинга через DoH.
#[derive(Serialize)]
struct DohResult {
    hostname: String,
    provider: String,
    ips: Vec<String>,
}

/// Резолвит домен через указанный DNS-over-HTTPS провайдера.
#[command]
async fn doh_resolve(hostname: String, provider: String) -> Result<DohResult, String> {
    let doh_urls = HashMap::from([
        ("cloudflare", "https://cloudflare-dns.com/dns-query?name={}&type=A"),
        ("google", "https://dns.google/resolve?name={}&type=A"),
        ("quad9", "https://dns.quad9.net/dns-query?name={}&type=A"),
        ("adguard", "https://dns.adguard.com/dns-query?name={}&type=A"),
    ]);

    let url_template = doh_urls
        .get(&provider.as_str())
        .ok_or_else(|| format!("unknown DoH provider: {}", provider))?;

    let url = url_template.replace("{}", &hostname);

    let client = reqwest::Client::new();
    let resp = client
        .get(&url)
        .header("accept", "application/dns-json")
        .send()
        .await
        .map_err(|e| format!("DoH request failed: {}", e))?;

    if !resp.status().is_success() {
        return Err(format!("DoH provider returned status {}", resp.status()));
    }

    let json: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("DoH JSON parse error: {}", e))?;

    let mut ips = Vec::new();
    if let Some(answer) = json.get("Answer").and_then(|a| a.as_array()) {
        for entry in answer {
            if entry.get("type").and_then(|t| t.as_u64()) == Some(1) {
                if let Some(data) = entry.get("data").and_then(|d| d.as_str()) {
                    ips.push(data.to_string());
                }
            }
        }
    }

    Ok(DohResult { hostname, provider, ips })
}

/// Возвращает статистику блокировок за текущую сессию из активной вкладки.
#[command]
async fn get_privacy_stats(webview: Webview) -> Result<String, String> {
    if let Some(fresh) = snapshot_page_stats(webview).await {
        if let Ok(mut guard) = PAGE_STATS.lock() {
            *guard = fresh.clone();
        }
        return Ok(fresh);
    }
    Ok(PAGE_STATS
        .lock()
        .map(|g| g.clone())
        .unwrap_or_else(|_| "{}".to_string()))
}

/// Обновляет список фильтров из собственного репозитория.
///
/// Раньше функция скачивала `keywords.dat` (список цензуры апстрима) и
/// возвращала заведомо пустые `blockedDomains`/`blockedUrlPatterns`, поэтому
/// «Обновить список фильтров» стирало блокировку вместо обновления.
/// Теперь берётся filter-rules.json и валидируется перед возвратом.
#[command]
async fn update_filter_list() -> Result<String, String> {
    const SOURCES: [&str; 2] = [
        "https://raw.githubusercontent.com/oekyaroslavl-sketch/browser/main/public/filter-rules.json",
        "https://raw.githubusercontent.com/oekyaroslavl-sketch/browser/arena/72386f61-browser/public/filter-rules.json",
    ];

    let client = reqwest::Client::builder()
        .user_agent("DiFri-Privacy/2.1")
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .map_err(|e| format!("cannot create HTTP client: {}", e))?;

    let mut last_error = String::from("no sources configured");
    for url in SOURCES {
        match client.get(url).send().await {
            Ok(resp) if resp.status().is_success() => match resp.text().await {
                Ok(text) => match serde_json::from_str::<serde_json::Value>(&text) {
                    Ok(v) if validate_rules(&v) => {
                        return serde_json::to_string(&v)
                            .map_err(|e| format!("cannot serialize rules: {}", e));
                    }
                    Ok(_) => last_error = format!("{} returned invalid rules", url),
                    Err(e) => last_error = format!("{} returned malformed JSON: {}", url, e),
                },
                Err(e) => last_error = format!("{} read error: {}", url, e),
            },
            Ok(resp) => last_error = format!("{} returned status {}", url, resp.status()),
            Err(e) => last_error = format!("{} request failed: {}", url, e),
        }
    }

    Err(last_error)
}

/// Минимальная проверка структуры правил перед применением.
fn validate_rules(v: &serde_json::Value) -> bool {
    let domains = v
        .get("blockedDomains")
        .and_then(|d| d.as_array())
        .map(|a| a.iter().filter(|x| x.as_str().is_some()).count())
        .unwrap_or(0);
    // Пустой список хуже, чем ничего: он отключил бы блокировку.
    domains > 0
}

/// Параметры навигации для конкретной вкладки.
/// Создать новую вкладку (дочерний вебвью) в главном окне.
/// x, y — смещение относительно окна (чтобы пропустить панель управления и табстрип).
/// РќР°РІРёРіР°С†РёСЏ РµРґРёРЅСЃС‚РІРµРЅРЅРѕРіРѕ РІРµР±РІСЊСЋ РЅР° URL СЃ РёРЅСЉРµРєС†РёРµР№ РєРѕРЅС‚РµРЅС‚-СЃРєСЂРёРїС‚Р°.
/// JS РІС‹Р·С‹РІР°РµС‚ РµС‘ СЃ РїР»РѕСЃРєРёРјРё РїР°СЂР°РјРµС‚СЂР°РјРё; webview Рё app РІРЅРµРґСЂСЏСЋС‚СЃСЏ Tauri.
#[command]
fn navigate_tab(
    webview: Webview,
    label: String,
    url: String,
    app_origin: String,
    doh_enabled: Option<bool>,
    doh_provider: Option<String>,
    permissions_denied: Option<bool>,
    block_third_party_cookies: Option<bool>,
    https_only: Option<bool>,
    rules_json: Option<String>,
    cosmetic_blocked: Option<bool>,
) -> Result<(), String> {
    // Правила: приоритет у переданных из UI, иначе — встроенный файл.
    // До этого `{}` попадал в контент-скрипт и затирал его встроенный RULES
    // пустым объектом — блокировка переставала работать после первого перехода.
    let rules = match rules_json {
        Some(r) if r.trim().len() > 2 && r.trim() != "{}" => r,
        _ => include_str!("../../public/filter-rules.json").to_string(),
    };

    let cfg = InjectionConfig {
        app_origin: app_origin.clone(),
        label: label.clone(),
        doh_enabled: doh_enabled.unwrap_or(false),
        doh_provider: doh_provider.unwrap_or_else(|| "cloudflare".to_string()),
        permissions_denied: permissions_denied.unwrap_or(false),
        block_third_party_cookies: block_third_party_cookies.unwrap_or(false),
        https_only: https_only.unwrap_or(false),
        rules_json: rules,
        cosmetic_blocked: cosmetic_blocked.unwrap_or(false),
    };

    // Запоминаем настройки последнего перехода (используются для снимка статистики).
    if let Ok(mut guard) = INJECTION.lock() {
        *guard = Some(cfg.clone());
    }
    // Новое поколение навигации: фоновые задачи предыдущего перехода
    // больше не должны вводить свои настройки в новый документ.
    let generation = {
        let mut g = NAV_GENERATION.lock().map_err(|_| "navigation lock poisoned".to_string())?;
        *g += 1;
        *g
    };

    let target = normalize_url(&url)?;

    // Если это внутренняя страница приложения — переключаемся на неё без инъекции
    // контент-скрипта на сам app.html (иначе панель встроится в интерфейс браузера).
    let is_app_page = cfg.app_origin != "null"
        && !cfg.app_origin.is_empty()
        && target.starts_with(&cfg.app_origin);

    let parsed: tauri::Url = target
        .parse()
        .map_err(|e| format!("invalid URL: {}", e))?;

    if is_app_page {
        if let Ok(mut guard) = INJECTION.lock() {
            *guard = None;
        }
    }

    // Навигация. Ошибку не пробрасываем наверх: на Windows неудачный переход
    // раньше оставлял UI в подвешенном состоянии, потому что JS ждал промис.
    if let Err(e) = webview.navigate(parsed.clone()) {
        eprintln!("[difri] navigate failed: {e}");
        show_error_page(&webview, &target);
        return Ok(());
    }

    // Инъекция контент-скрипта.
    //
    // В tauri 2.12 нет ни `Webview::on_page_load`, ни `WebviewWindow::on_page_load`
    // (проект собирается с feature `unstable`, где webview и window разделены),
    // поэтому вместо хука загрузки — серия идемпотентных попыток `eval`.
    // Guard `window.__difriInjected` внутри скрипта делает повторные вызовы
    // безопасными: пока документ не заменён навигацией, скрипт вводится один раз,
    // а в новом документе window пустой и инъекция повторяется.
    if !is_app_page {
        let wv = webview.clone();
        let cfg2 = cfg.clone();
        tauri::async_runtime::spawn(async move {
            for delay_ms in [200u64, 500, 900, 1500, 2500] {
                tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                let current = NAV_GENERATION.lock().map(|g| *g).unwrap_or_default();
                if current != generation {
                    return; // пользователь уже ушёл на другую страницу
                }
                let _ = wv.eval(&build_content_script(&cfg2));
                // Снимок статистики заберёт отчёт о приватности по запросу.
            }
        });
    }

    Ok(())
}

/// Переходит на внутреннюю страницу ошибки, если сайт открыть не удалось.
fn show_error_page(webview: &Webview, target: &str) {
    let encoded = percent_encode(target);
    let _ = webview.eval(&format!(
        "window.__difriFailedUrl = {};",
        serde_json::to_string(target).unwrap_or_else(|_| "\"\"".to_string())
    ));
    // Внутренние страницы живут по tauri://localhost/ (Windows) или
    // http://tauri.localhost/ — точный адрес отдаёт сам WebView2.
    if let Ok(base) = webview.url() {
        let origin = format!(
            "{}://{}",
            base.scheme(),
            base.host_str().unwrap_or("tauri.localhost")
        );
        let path = base.path().rsplit_once('/').map(|(p, _)| p).unwrap_or("");
        if let Ok(url) = format!("{}{}/error.html#{}", origin, path, encoded).parse::<tauri::Url>() {
            let _ = webview.navigate(url);
        }
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_shell::init())
        .manage(DownloadManager::default())
        .invoke_handler(generate_handler![
            clear_browsing_data,
            app_version,
            navigate_tab,
            doh_resolve,
            get_privacy_stats,
            download_start,
            download_cancel,
            download_open,
            get_downloads,
            update_filter_list
        ])
        .setup(|_app| {
            // Инъекция контент-скрипта выполняется в navigate_tab:
            // API on_page_load в tauri 2.12 (feature `unstable`) отсутствует.
            Ok(())
        })
        .run(generate_context!())
        .expect("error while running DiFri Privacy");
}
