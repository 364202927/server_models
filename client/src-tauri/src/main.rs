#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::fs;
use std::path::PathBuf;

#[cfg(windows)]
use std::os::windows::process::CommandExt;
#[cfg(windows)]
use std::process::Command;

// 前端允许读写的配置文件白名单，防止传入任意路径
const CONFIG_FILES: [&str; 2] = ["modelSetting.json", "models.json"];

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;
#[cfg(windows)]
const WSL_DISTRO: &str = "Ubuntu";
#[cfg(windows)]
const CONDA_ENV: &str = "work";
// Windows 下的项目根目录（用于直接读写 assets/*.json）
#[cfg(windows)]
const PROJECT_WIN_PATH: &str = r"D:\server_models";
// WSL 内部对应的项目路径
#[cfg(windows)]
const PROJECT_WSL_PATH: &str = "/mnt/d/server_models";

/// Windows 下 assets 位于项目目录；其他平台（如 Mac）通过环境变量指向本地 assets 目录
#[cfg(windows)]
fn assets_dir() -> Result<PathBuf, String> {
    Ok(PathBuf::from(PROJECT_WIN_PATH).join("assets"))
}

#[cfg(not(windows))]
fn assets_dir() -> Result<PathBuf, String> {
    std::env::var("SERVER_MODELS_ASSETS_DIR")
        .map(PathBuf::from)
        .map_err(|_| "非 Windows 平台请设置环境变量 SERVER_MODELS_ASSETS_DIR 指向 assets 目录".into())
}

fn config_path(filename: &str) -> Result<PathBuf, String> {
    if !CONFIG_FILES.contains(&filename) {
        return Err(format!("不允许访问的配置文件: {}", filename));
    }
    Ok(assets_dir()?.join(filename))
}

#[cfg(windows)]
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

#[cfg(not(windows))]
#[tauri::command]
fn start_wsl_backend() -> Result<String, String> {
    Err("仅 Windows 支持拉起 WSL 后端，当前平台将直接连接网关".into())
}

#[cfg(windows)]
#[tauri::command]
fn stop_wsl_backend() -> Result<String, String> {
    Command::new("wsl.exe")
        .args(["--terminate", WSL_DISTRO])
        .creation_flags(CREATE_NO_WINDOW)
        .output()
        .map_err(|e| format!("关闭 WSL 失败: {}", e))?;

    Ok("terminated".into())
}

#[cfg(not(windows))]
#[tauri::command]
fn stop_wsl_backend() -> Result<String, String> {
    Err("仅 Windows 支持关停 WSL 后端".into())
}

#[tauri::command]
fn read_json_config(filename: String) -> Result<String, String> {
    let path = config_path(&filename)?;
    fs::read_to_string(&path).map_err(|e| format!("读取文件失败 {:?}: {}", path, e))
}

#[tauri::command]
fn write_json_config(filename: String, content: String) -> Result<(), String> {
    let target = config_path(&filename)?;
    let temp = target.with_extension("json.tmp");

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
