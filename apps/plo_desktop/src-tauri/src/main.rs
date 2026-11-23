// Prevents an extra console window on Windows in release builds
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::Arc;

use plo_desktop_commands::{dispatch_command, DesktopState, Json, SpotConfig};
use tauri::{AppHandle, Emitter, State};

fn serde_to_our_json(v: &serde_json::Value) -> Json {
    Json::parse(&v.to_string()).unwrap_or(Json::Null)
}

fn our_json_to_serde(j: &Json) -> serde_json::Value {
    serde_json::from_str(&j.to_json_string()).unwrap_or(serde_json::Value::Null)
}

fn run_sync_cmd(
    state: &Arc<DesktopState>,
    cmd: &str,
    args: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let j_args = serde_to_our_json(&args);
    let out = dispatch_command(state, cmd, &j_args)?;
    Ok(our_json_to_serde(&out))
}

#[tauri::command]
fn list_spots(state: State<'_, Arc<DesktopState>>) -> Result<serde_json::Value, String> {
    Ok(our_json_to_serde(&state.list_spots()))
}

#[tauri::command]
fn set_pack_folder(
    state: State<'_, Arc<DesktopState>>,
    args: serde_json::Value,
) -> Result<serde_json::Value, String> {
    run_sync_cmd(&state, "set_pack_folder", args)
}

#[tauri::command]
fn bind_spot(
    state: State<'_, Arc<DesktopState>>,
    args: serde_json::Value,
) -> Result<serde_json::Value, String> {
    run_sync_cmd(&state, "bind_spot", args)
}

#[tauri::command]
fn default_spot_for_seat(
    state: State<'_, Arc<DesktopState>>,
    args: serde_json::Value,
) -> Result<serde_json::Value, String> {
    run_sync_cmd(&state, "default_spot_for_seat", args)
}

#[tauri::command]
fn select_spot(
    state: State<'_, Arc<DesktopState>>,
    args: serde_json::Value,
) -> Result<serde_json::Value, String> {
    run_sync_cmd(&state, "select_spot", args)
}

#[tauri::command]
fn set_memory_budget(
    state: State<'_, Arc<DesktopState>>,
    args: serde_json::Value,
) -> Result<serde_json::Value, String> {
    run_sync_cmd(&state, "set_memory_budget", args)
}

#[tauri::command]
fn villain_from_spot(
    state: State<'_, Arc<DesktopState>>,
    args: serde_json::Value,
) -> Result<serde_json::Value, String> {
    run_sync_cmd(&state, "villain_from_spot", args)
}

#[tauri::command]
fn preview_villain(
    state: State<'_, Arc<DesktopState>>,
    args: serde_json::Value,
) -> Result<serde_json::Value, String> {
    run_sync_cmd(&state, "preview_villain", args)
}

#[tauri::command]
fn parse_hero(
    state: State<'_, Arc<DesktopState>>,
    args: serde_json::Value,
) -> Result<serde_json::Value, String> {
    run_sync_cmd(&state, "parse_hero", args)
}

#[tauri::command]
fn parse_board(
    state: State<'_, Arc<DesktopState>>,
    args: serde_json::Value,
) -> Result<serde_json::Value, String> {
    run_sync_cmd(&state, "parse_board", args)
}

#[tauri::command]
fn compute_equity(
    state: State<'_, Arc<DesktopState>>,
    args: serde_json::Value,
) -> Result<serde_json::Value, String> {
    run_sync_cmd(&state, "compute_equity", args)
}

#[tauri::command]
fn estimate_spot(
    state: State<'_, Arc<DesktopState>>,
    args: serde_json::Value,
) -> Result<serde_json::Value, String> {
    run_sync_cmd(&state, "estimate_spot", args)
}

#[tauri::command]
fn build_game(
    state: State<'_, Arc<DesktopState>>,
    args: serde_json::Value,
) -> Result<serde_json::Value, String> {
    run_sync_cmd(&state, "build_game", args)
}

#[tauri::command]
async fn solve_spot(
    app: AppHandle,
    state: State<'_, Arc<DesktopState>>,
    args: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let j_args = serde_to_our_json(&args);
    let cfg = SpotConfig::from_json(&j_args)?;
    let shared = Arc::clone(&state);

    // Run DCFR on a dedicated OS worker thread so the UI thread stays 100% responsive
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let res = shared.solve_blocking(&cfg, 0, |ev| {
            let payload = our_json_to_serde(&ev.to_json());
            let _ = app.emit("solve-progress", payload);
        });
        let _ = tx.send(res);
    });

    let res = tauri::async_runtime::spawn_blocking(move || {
        rx.recv().map_err(|_| "solver worker thread disconnected".to_string())?
    })
    .await
    .map_err(|e| format!("task join error: {e}"))??;

    Ok(our_json_to_serde(&res))
}

#[tauri::command]
fn stop_solve(state: State<'_, Arc<DesktopState>>) -> Result<serde_json::Value, String> {
    state.stop_solve();
    Ok(serde_json::json!({ "stopped": true }))
}

#[tauri::command]
fn latest_progress(state: State<'_, Arc<DesktopState>>) -> Result<serde_json::Value, String> {
    run_sync_cmd(&state, "latest_progress", serde_json::json!({}))
}

#[tauri::command]
async fn step_iterations(
    state: State<'_, Arc<DesktopState>>,
    args: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let shared = Arc::clone(&state);
    let j_args = serde_to_our_json(&args);
    let out = tauri::async_runtime::spawn_blocking(move || {
        dispatch_command(&shared, "step_iterations", &j_args)
    })
    .await
    .map_err(|e| e.to_string())??;
    Ok(our_json_to_serde(&out))
}

#[tauri::command]
fn query_node(state: State<'_, Arc<DesktopState>>) -> Result<serde_json::Value, String> {
    run_sync_cmd(&state, "query_node", serde_json::json!({}))
}

#[tauri::command]
fn set_villain_filter(
    state: State<'_, Arc<DesktopState>>,
    args: serde_json::Value,
) -> Result<serde_json::Value, String> {
    run_sync_cmd(&state, "set_villain_filter", args)
}

#[tauri::command]
async fn lock_node(
    state: State<'_, Arc<DesktopState>>,
    args: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let shared = Arc::clone(&state);
    let j_args = serde_to_our_json(&args);
    let out = tauri::async_runtime::spawn_blocking(move || {
        dispatch_command(&shared, "lock_node", &j_args)
    })
    .await
    .map_err(|e| e.to_string())??;
    Ok(our_json_to_serde(&out))
}

#[tauri::command]
async fn unlock_node(
    state: State<'_, Arc<DesktopState>>,
    args: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let shared = Arc::clone(&state);
    let j_args = serde_to_our_json(&args);
    let out = tauri::async_runtime::spawn_blocking(move || {
        dispatch_command(&shared, "unlock_node", &j_args)
    })
    .await
    .map_err(|e| e.to_string())??;
    Ok(our_json_to_serde(&out))
}

#[tauri::command]
fn play_action(
    state: State<'_, Arc<DesktopState>>,
    args: serde_json::Value,
) -> Result<serde_json::Value, String> {
    run_sync_cmd(&state, "play_action", args)
}

#[tauri::command]
fn play_chance(
    state: State<'_, Arc<DesktopState>>,
    args: serde_json::Value,
) -> Result<serde_json::Value, String> {
    run_sync_cmd(&state, "play_chance", args)
}

#[tauri::command]
fn rewind_to(
    state: State<'_, Arc<DesktopState>>,
    args: serde_json::Value,
) -> Result<serde_json::Value, String> {
    run_sync_cmd(&state, "rewind_to", args)
}

#[tauri::command]
fn reset_cursor(state: State<'_, Arc<DesktopState>>) -> Result<serde_json::Value, String> {
    run_sync_cmd(&state, "reset_cursor", serde_json::json!({}))
}

#[tauri::command]
async fn resolve_subgame(
    state: State<'_, Arc<DesktopState>>,
    args: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let shared = Arc::clone(&state);
    let j_args = serde_to_our_json(&args);
    let out = tauri::async_runtime::spawn_blocking(move || {
        dispatch_command(&shared, "resolve_subgame", &j_args)
    })
    .await
    .map_err(|e| e.to_string())??;
    Ok(our_json_to_serde(&out))
}

#[tauri::command]
fn exit_subgame(state: State<'_, Arc<DesktopState>>) -> Result<serde_json::Value, String> {
    run_sync_cmd(&state, "exit_subgame", serde_json::json!({}))
}

#[tauri::command]
async fn lock_node_by_classes(
    state: State<'_, Arc<DesktopState>>,
    args: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let shared = Arc::clone(&state);
    let j_args = serde_to_our_json(&args);
    let out = tauri::async_runtime::spawn_blocking(move || {
        dispatch_command(&shared, "lock_node_by_classes", &j_args)
    })
    .await
    .map_err(|e| e.to_string())??;
    Ok(our_json_to_serde(&out))
}

#[tauri::command]
fn jump_street_path(
    state: State<'_, Arc<DesktopState>>,
    args: serde_json::Value,
) -> Result<serde_json::Value, String> {
    run_sync_cmd(&state, "jump_street_path", args)
}

#[tauri::command]
fn list_saved_spots(state: State<'_, Arc<DesktopState>>) -> Result<serde_json::Value, String> {
    run_sync_cmd(&state, "list_saved_spots", serde_json::json!({}))
}

#[tauri::command]
fn save_spot_bookmark(
    state: State<'_, Arc<DesktopState>>,
    args: serde_json::Value,
) -> Result<serde_json::Value, String> {
    run_sync_cmd(&state, "save_spot_bookmark", args)
}

#[tauri::command]
fn delete_saved_spot(
    state: State<'_, Arc<DesktopState>>,
    args: serde_json::Value,
) -> Result<serde_json::Value, String> {
    run_sync_cmd(&state, "delete_saved_spot", args)
}

#[tauri::command]
fn export_node_report(state: State<'_, Arc<DesktopState>>) -> Result<serde_json::Value, String> {
    run_sync_cmd(&state, "export_node_report", serde_json::json!({}))
}

#[tauri::command]
fn parse_hand_history(
    state: State<'_, Arc<DesktopState>>,
    args: serde_json::Value,
) -> Result<serde_json::Value, String> {
    run_sync_cmd(&state, "parse_hand_history", args)
}

#[tauri::command]
async fn run_flop_subset_report(
    state: State<'_, Arc<DesktopState>>,
    args: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let shared = Arc::clone(&state);
    let j_args = serde_to_our_json(&args);
    let out = tauri::async_runtime::spawn_blocking(move || {
        dispatch_command(&shared, "run_flop_subset_report", &j_args)
    })
    .await
    .map_err(|e| e.to_string())??;
    Ok(our_json_to_serde(&out))
}

#[tauri::command]
async fn audit_hand_history(
    state: State<'_, Arc<DesktopState>>,
    args: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let shared = Arc::clone(&state);
    let j_args = serde_to_our_json(&args);
    let out = tauri::async_runtime::spawn_blocking(move || {
        dispatch_command(&shared, "audit_hand_history", &j_args)
    })
    .await
    .map_err(|e| e.to_string())??;
    Ok(our_json_to_serde(&out))
}

#[tauri::command]
fn start_bot_hand(
    state: State<'_, Arc<DesktopState>>,
    args: serde_json::Value,
) -> Result<serde_json::Value, String> {
    run_sync_cmd(&state, "start_bot_hand", args)
}

#[tauri::command]
fn bot_play_hero_action(
    state: State<'_, Arc<DesktopState>>,
    args: serde_json::Value,
) -> Result<serde_json::Value, String> {
    run_sync_cmd(&state, "bot_play_hero_action", args)
}

#[tauri::command]
fn stop_bot_hand(state: State<'_, Arc<DesktopState>>) -> Result<serde_json::Value, String> {
    run_sync_cmd(&state, "stop_bot_hand", serde_json::json!({}))
}

#[tauri::command]
async fn compare_bet_simplifications(
    state: State<'_, Arc<DesktopState>>,
    args: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let shared = Arc::clone(&state);
    let j_args = serde_to_our_json(&args);
    let out = tauri::async_runtime::spawn_blocking(move || {
        dispatch_command(&shared, "compare_bet_simplifications", &j_args)
    })
    .await
    .map_err(|e| e.to_string())??;
    Ok(our_json_to_serde(&out))
}

#[tauri::command]
fn save_tree_snapshot(
    state: State<'_, Arc<DesktopState>>,
    args: serde_json::Value,
) -> Result<serde_json::Value, String> {
    run_sync_cmd(&state, "save_tree_snapshot", args)
}

#[tauri::command]
fn load_tree_snapshot(
    state: State<'_, Arc<DesktopState>>,
    args: serde_json::Value,
) -> Result<serde_json::Value, String> {
    run_sync_cmd(&state, "load_tree_snapshot", args)
}

#[tauri::command]
fn export_config(
    state: State<'_, Arc<DesktopState>>,
    args: serde_json::Value,
) -> Result<serde_json::Value, String> {
    run_sync_cmd(&state, "export_config", args)
}

#[tauri::command]
fn import_config(
    state: State<'_, Arc<DesktopState>>,
    args: serde_json::Value,
) -> Result<serde_json::Value, String> {
    run_sync_cmd(&state, "import_config", args)
}

fn main() {
    let shared_state = Arc::new(DesktopState::default());

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(shared_state)
        .invoke_handler(tauri::generate_handler![
            list_spots,
            set_pack_folder,
            bind_spot,
            default_spot_for_seat,
            select_spot,
            set_memory_budget,
            villain_from_spot,
            preview_villain,
            parse_hero,
            parse_board,
            compute_equity,
            estimate_spot,
            build_game,
            solve_spot,
            stop_solve,
            latest_progress,
            step_iterations,
            query_node,
            set_villain_filter,
            lock_node,
            unlock_node,
            lock_node_by_classes,
            play_action,
            play_chance,
            rewind_to,
            reset_cursor,
            resolve_subgame,
            exit_subgame,
            jump_street_path,
            list_saved_spots,
            save_spot_bookmark,
            delete_saved_spot,
            export_node_report,
            parse_hand_history,
            run_flop_subset_report,
            audit_hand_history,
            start_bot_hand,
            bot_play_hero_action,
            stop_bot_hand,
            compare_bet_simplifications,
            save_tree_snapshot,
            load_tree_snapshot,
            export_config,
            import_config,
        ])
        .run(tauri::generate_context!())
        .expect("error while running PLO-4 Postflop — Hero vs Range");
}
