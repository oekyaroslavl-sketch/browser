//! Zero-dependency native HTTP + SSE bridge serving `apps/plo_desktop/ui/` on `0.0.0.0:4173`
//! (or `$PORT`) and executing all `plo_desktop_commands` natively in Rust.
//!
//! Used for instant live preview & headless Linux testing where `webkit2gtk` system packages
//! are absent, while `apps/plo_desktop/src-tauri` builds the native Windows/macOS/Linux Tauri 2 shell.

use std::env;
use std::fs;
use std::io::{BufRead, BufReader, ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::Duration;

use plo_desktop_commands::{dispatch_command, DesktopState, Json, SpotConfig};

struct SseBroadcaster {
    clients: Mutex<Vec<mpsc::Sender<String>>>,
}

impl SseBroadcaster {
    fn new() -> Self {
        Self {
            clients: Mutex::new(Vec::new()),
        }
    }

    fn subscribe(&self) -> mpsc::Receiver<String> {
        let (tx, rx) = mpsc::channel();
        self.clients.lock().unwrap().push(tx);
        rx
    }

    fn broadcast_event(&self, event_name: &str, json_data: &str) {
        let msg = format!("event: {event_name}\ndata: {json_data}\n\n");
        let mut guard = self.clients.lock().unwrap();
        guard.retain(|tx| tx.send(msg.clone()).is_ok());
    }
}

fn find_ui_dir() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if let Some(root) = manifest_dir.parent().and_then(|p| p.parent()) {
        let candidate = root.join("apps").join("plo_desktop").join("ui");
        if candidate.exists() {
            return candidate;
        }
    }
    PathBuf::from("apps/plo_desktop/ui")
}

fn main() {
    let port: u16 = env::var("PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4173);
    let ui_dir = find_ui_dir();
    let state = Arc::new(DesktopState::default());

    // If no pack is saved yet and `.local/nandez_pack` exists locally, auto-load it for convenience
    if state.list_spots().get("loaded").and_then(|v| v.as_bool()) != Some(true) {
        let local_pack = Path::new(".local/nandez_pack");
        if local_pack.exists() {
            let _ = state.set_pack_folder(local_pack);
        }
    }

    let sse = Arc::new(SseBroadcaster::new());
    let addr = format!("0.0.0.0:{port}");
    let listener = TcpListener::bind(&addr).expect("failed to bind preview server");
    println!(
        "PLO-4 Postflop — Hero vs Range UI bridge listening on http://{addr} (serving {})",
        ui_dir.display()
    );

    for stream in listener.incoming().flatten() {
        let st = Arc::clone(&state);
        let br = Arc::clone(&sse);
        let dir = ui_dir.clone();
        thread::spawn(move || {
            let _ = handle_connection(stream, &dir, &st, &br);
        });
    }
}

fn handle_connection(
    mut stream: TcpStream,
    ui_dir: &Path,
    state: &Arc<DesktopState>,
    sse: &Arc<SseBroadcaster>,
) -> std::io::Result<()> {
    let _ = stream.set_nodelay(true);
    let _ = stream.set_read_timeout(Some(Duration::from_secs(30)));
    let mut reader = BufReader::new(stream.try_clone()?);

    // Support HTTP/1.1 keep-alive connection reuse across reverse proxies (e.g. e2b.app)
    loop {
        let mut req_line = String::new();
        match reader.read_line(&mut req_line) {
            Ok(0) => return Ok(()),
            Ok(_) => {}
            Err(e) if e.kind() == ErrorKind::WouldBlock || e.kind() == ErrorKind::TimedOut => {
                return Ok(());
            }
            Err(e) => return Err(e),
        }
        if req_line.trim().is_empty() {
            continue;
        }
        let parts: Vec<&str> = req_line.split_whitespace().collect();
        if parts.len() < 2 {
            return Ok(());
        }
        let method = parts[0];
        let raw_path = parts[1].split('?').next().unwrap_or("/");

        let mut content_len = 0usize;
        let mut client_wants_close = false;
        loop {
            let mut hdr = String::new();
            if reader.read_line(&mut hdr)? == 0 || hdr == "\r\n" || hdr == "\n" {
                break;
            }
            if let Some((k, v)) = hdr.split_once(':') {
                let key = k.trim();
                let val = v.trim();
                if key.eq_ignore_ascii_case("content-length") {
                    content_len = val.parse().unwrap_or(0);
                } else if key.eq_ignore_ascii_case("connection")
                    && val.to_ascii_lowercase().contains("close")
                {
                    client_wants_close = true;
                }
            }
        }

        let mut body = vec![0u8; content_len];
        if content_len > 0 {
            reader.read_exact(&mut body)?;
        }

        let conn_hdr = if client_wants_close { "close" } else { "keep-alive" };

        if method == "OPTIONS" {
            let resp = format!(
                "HTTP/1.1 204 No Content\r\nContent-Length: 0\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, HEAD, POST, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type\r\nConnection: {conn_hdr}\r\n\r\n"
            );
            stream.write_all(resp.as_bytes())?;
            stream.flush()?;
            if client_wants_close {
                return Ok(());
            }
            continue;
        }

        if method == "GET" && (raw_path == "/api/events" || raw_path.ends_with("/api/events")) {
            let _ = stream.set_read_timeout(None);
            let hdr = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nX-Accel-Buffering: no\r\nAccess-Control-Allow-Origin: *\r\n\r\n: connected\n\n";
            stream.write_all(hdr.as_bytes())?;
            stream.flush()?;
            let rx = sse.subscribe();
            loop {
                match rx.recv_timeout(Duration::from_secs(15)) {
                    Ok(msg) => {
                        if stream.write_all(msg.as_bytes()).is_err() || stream.flush().is_err() {
                            break;
                        }
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        if stream.write_all(b": ping\n\n").is_err() || stream.flush().is_err() {
                            break;
                        }
                    }
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                }
            }
            return Ok(());
        }

        if method == "POST" && (raw_path == "/api/cmd" || raw_path.ends_with("/api/cmd")) {
            let body_str = String::from_utf8_lossy(&body);
            let reply = match Json::parse(&body_str) {
                Ok(req_json) => {
                    let cmd = req_json.get("cmd").and_then(|v| v.as_str()).unwrap_or("");
                    let args = req_json.get("args").cloned().unwrap_or_else(|| Json::obj([]));
                    let result = if cmd == "solve_spot" {
                        match SpotConfig::from_json(&args) {
                            Ok(cfg) => state.solve_blocking(&cfg, 0, |ev| {
                                sse.broadcast_event("solve-progress", &ev.to_json().to_json_string());
                            }),
                            Err(e) => Err(e),
                        }
                    } else {
                        dispatch_command(state, cmd, &args)
                    };
                    match result {
                        Ok(data) => Json::obj([("ok", Json::Bool(true)), ("data", data)]),
                        Err(err) => Json::obj([("ok", Json::Bool(false)), ("error", Json::Str(err))]),
                    }
                }
                Err(e) => Json::obj([
                    ("ok", Json::Bool(false)),
                    ("error", Json::Str(format!("invalid JSON body: {e}"))),
                ]),
            };

            let out_bytes = reply.to_json_string().into_bytes();
            let hdr = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nConnection: {conn_hdr}\r\n\r\n",
                out_bytes.len()
            );
            stream.write_all(hdr.as_bytes())?;
            stream.write_all(&out_bytes)?;
            stream.flush()?;
            if client_wants_close {
                return Ok(());
            }
            continue;
        }

        if raw_path == "/favicon.ico" {
            let hdr = format!(
                "HTTP/1.1 204 No Content\r\nContent-Length: 0\r\nConnection: {conn_hdr}\r\n\r\n"
            );
            stream.write_all(hdr.as_bytes())?;
            stream.flush()?;
            if client_wants_close {
                return Ok(());
            }
            continue;
        }

        // Serve static files from `apps/plo_desktop/ui` (support both GET and HEAD)
        let rel = match raw_path {
            "/" | "/index.html" => "index.html",
            "/styles.css" => "styles.css",
            "/app.js" => "app.js",
            other => other.trim_start_matches('/'),
        };
        if rel.contains("..") {
            let resp = format!(
                "HTTP/1.1 403 Forbidden\r\nContent-Length: 9\r\nConnection: {conn_hdr}\r\n\r\nForbidden"
            );
            stream.write_all(resp.as_bytes())?;
            stream.flush()?;
            if client_wants_close {
                return Ok(());
            }
            continue;
        }
        let file_path = ui_dir.join(rel);
        if let Ok(bytes) = fs::read(&file_path) {
            let mime = if rel.ends_with(".html") {
                "text/html; charset=utf-8"
            } else if rel.ends_with(".css") {
                "text/css; charset=utf-8"
            } else if rel.ends_with(".js") {
                "application/javascript; charset=utf-8"
            } else {
                "application/octet-stream"
            };
            let hdr = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: {mime}\r\nContent-Length: {}\r\nCache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\nConnection: {conn_hdr}\r\n\r\n",
                bytes.len()
            );
            stream.write_all(hdr.as_bytes())?;
            if method != "HEAD" {
                stream.write_all(&bytes)?;
            }
            stream.flush()?;
        } else {
            let resp = if method == "HEAD" {
                format!("HTTP/1.1 404 Not Found\r\nContent-Length: 9\r\nConnection: {conn_hdr}\r\n\r\n")
            } else {
                format!("HTTP/1.1 404 Not Found\r\nContent-Length: 9\r\nConnection: {conn_hdr}\r\n\r\nNot Found")
            };
            stream.write_all(resp.as_bytes())?;
            stream.flush()?;
        }
        if client_wants_close {
            return Ok(());
        }
    }
}
