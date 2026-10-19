#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::os::windows::process::CommandExt;

const CREATE_NO_WINDOW: u32 = 0x08000000;
const WSL_DISTRO: &str = "Ubuntu";
const CONDA_ENV: &str = "base";
// Windows 下的项目根目录（用于直接读写 assets/*.json）
const PROJECT_WIN_PATH: &str = r"D:\server_models";
// WSL 内部对应的项目路径
const PROJECT_WSL_PATH: &str = "/mnt/d/server_models";

#[tauri::command]
fn start_wsl_backend() -> Result<String, String> {
    let cmd = format!(
        "conda activate {} && cd {} && python main.py",
        CONDA_ENV, PROJECT_WSL_PATH
    );
    Command::new("wsl.exe")
        .args(["-d", WSL_DISTRO, "--", "bash", "-lic", &cmd])
        .creation_flags(CREATE_NO_WINDOW)
        .spawn()
        .map_err(|e| format!("拉起 WSL 失败: {}", e))?;

    Ok("spawned".into())
}

#[tauri::command]
fn stop_wsl_backend() -> Result<String, String> {
    Command::new("wsl.exe")
        .args(["--terminate", WSL_DISTRO])
        .creation_flags(CREATE_NO_WINDOW)
        .output()
        .map_err(|e| format!("关闭 WSL 失败: {}", e))?;

    Ok("terminated".into())
}

#[tauri::command]
fn read_json_config(filename: String) -> Result<String, String> {
    let path = PathBuf::from(PROJECT_WIN_PATH).join("assets").join(filename);
    fs::read_to_string(&path).map_err(|e| format!("读取文件失败 {:?}: {}", path, e))
}

#[tauri::command]
fn write_json_config(filename: String, content: String) -> Result<(), String> {
    let dir = PathBuf::from(PROJECT_WIN_PATH).join("assets");
    let target = dir.join(&filename);
    let temp = dir.join(format!("{}.tmp", filename));

    // 原子写入：先写临时文件再覆盖，防止写坏 JSON
    fs::write(&temp, content).map_err(|e| format!("写入临时文件失败: {}", e))?;
    fs::rename(&temp, &target).map_err(|e| format!("覆盖配置文件失败: {}", e))
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            start_wsl_backend,
            stop_wsl_backend,
            read_json_config,
            write_json_config
        ])
        .run(tauri::generate_context!())
        .expect("Tauri 运行异常");
}