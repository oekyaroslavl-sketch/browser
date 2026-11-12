#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};
use tauri::{command, generate_context, generate_handler, AppHandle, Manager, State, Webview};

/// Снимок статистики блокировок последней загруженной страницы (JSON).
static PAGE_STATS: Mutex<String> = Mutex::new(String::new());

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
        let mut total: Option<u64> = None;
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
async fn get_privacy_stats() -> Result<String, String> {
    Ok(PAGE_STATS.lock().map(|g| g.clone()).unwrap_or_default())
}

#[command]
async fn update_filter_list() -> Result<String, String> {
    let client = reqwest::Client::new();
    let url = "https://raw.githubusercontent.com/DiFri/DiFri-Python/Desktop/src/browser/keywords.dat";

    let resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| format!("Failed to fetch filter list: {}", e))?;

    if !resp.status().is_success() {
        return Err(format!("Filter list server returned status {}", resp.status()));
    }

    let _text = resp
        .text()
        .await
        .map_err(|e| format!("Failed to read filter list: {}", e))?;

    // Заготовка обновления правил: контент-скрипт принимает эту структуру
    // через window.__difriRulesUpdate. В production правила стоит публиковать
    // в формате filter-rules.json и раздавать из своего GitHub-репозитория.
    let update_json = serde_json::json!({
        "version": SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs().to_string())
            .unwrap_or_else(|_| "unknown".to_string()),
        "name": "DiFri Privacy Updated Filter List",
        "blockedDomains": [],
        "blockedUrlPatterns": [],
        "trackingQueryParams": [],
        "domainCategories": {}
    })
    .to_string();

    Ok(update_json)
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
    let parsed: tauri::Url = url.parse().map_err(|e| format!("invalid URL: {}", e))?;
    webview.navigate(parsed).map_err(|e| e.to_string())?;

    let doh_enabled = doh_enabled.unwrap_or(false);
    let doh_provider = doh_provider.unwrap_or_else(|| "cloudflare".to_string());
    let permissions_denied = permissions_denied.unwrap_or(false);
    let block_third_party_cookies = block_third_party_cookies.unwrap_or(false);
    let https_only = https_only.unwrap_or(false);
    let rules_json = rules_json.as_deref().unwrap_or("{}");
    let cosmetic_blocked = cosmetic_blocked.unwrap_or(false);

    let prefix = format!(
        "window.__difriAppOrigin = {};\nwindow.__difriDoh = {{ enabled: {}, provider: {} }};\nwindow.__difriPermissions = {{ denied: {} }};\nwindow.__difriCookies = {{ blockThirdPartyCookies: {} }};\nwindow.__difriHttps = {{ httpsOnly: {} }};\nwindow.__difriTabLabel = {};\nwindow.__difriRulesUpdate = {{ json: {} }};\nwindow.__difriCosmetic = {{ cosmeticBlocking: {} }};\n",
        serde_json::to_string(&app_origin).unwrap_or_else(|_| "\"\"".to_string()),
        serde_json::to_string(&doh_enabled).unwrap_or_else(|_| "false".to_string()),
        serde_json::to_string(&doh_provider).unwrap_or_else(|_| "\"cloudflare\"".to_string()),
        serde_json::to_string(&permissions_denied).unwrap_or_else(|_| "false".to_string()),
        serde_json::to_string(&block_third_party_cookies).unwrap_or_else(|_| "false".to_string()),
        serde_json::to_string(&https_only).unwrap_or_else(|_| "false".to_string()),
        serde_json::to_string(&label).unwrap_or_else(|_| "\"main\"".to_string()),
        rules_json,
        cosmetic_blocked
    );
    let script = format!("{}{}", prefix, include_str!("../../public/content-script.js"));

    let (tx, rx) = std::sync::mpsc::channel::<String>();
    webview
        .eval_with_callback(script, move |_res| {
            let _ = tx.send(String::new());
        })
        .map_err(|e| e.to_string())?;
    rx.recv_timeout(std::time::Duration::from_secs(5))
        .map_err(|e| format!("timeout: {}", e))?;

    // Р§РµСЂРµР· РЅРµСЃРєРѕР»СЊРєРѕ СЃРµРєСѓРЅРґ СЃРЅРёРјР°РµРј СЃС‚Р°С‚РёСЃС‚РёРєСѓ Р±Р»РѕРєРёСЂРѕРІРѕРє СЃРѕ СЃС‚СЂР°РЅРёС†С‹:
    // РєРѕРЅС‚РµРЅС‚-СЃРєСЂРёРїС‚ РїРёС€РµС‚ РµС‘ РІ sessionStorage РїСЂРѕРёСЃС…РѕР¶РґРµРЅРёСЏ СЃР°Р№С‚Р°.
    let wv = webview.clone();
    tauri::async_runtime::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_secs(4)).await;
        let (tx2, rx2) = std::sync::mpsc::channel::<String>();
        let snapshot = r#"(function(){ try { return { blockedCount: parseInt(sessionStorage.getItem('difri:cs:blocked')||'0',10), stats: JSON.parse(sessionStorage.getItem('difri:cs:stats')||'{}') }; } catch(e) { return {}; } })()"#;
        if wv
            .eval_with_callback(snapshot, move |res| {
                let _ = tx2.send(res);
            })
            .is_ok()
        {
            if let Ok(data) = rx2.recv_timeout(std::time::Duration::from_secs(5)) {
                if let Ok(mut guard) = PAGE_STATS.lock() {
                    *guard = data;
                }
            }
        }
    });

    Ok(())
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
            Ok(())
        })
        .run(generate_context!())
        .expect("error while running DiFri Privacy");
}
