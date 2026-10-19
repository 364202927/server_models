#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::fs;
use std::path::PathBuf;

#[cfg(windows)]
use std::os::windows::process::CommandExt;
#[cfg(windows)]
use std::process::{Command, Stdio};
use tauri::WebviewWindowBuilder;

// 前端允许读写的配置文件白名单，防止传入任意路径
const CONFIG_FILES: [&str; 2] = ["modelSetting.json", "models.json"];

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;
// creation_flags 会整体覆盖 Rust 默认的建进程标志，必须自己补回这一位，否则环境块按 ANSI 解析
#[cfg(windows)]
const CREATE_UNICODE_ENVIRONMENT: u32 = 0x0000_0400;
// Windows 下的项目根目录（用于直接读写 assets/*.json）
#[cfg(windows)]
const PROJECT_WIN_PATH: &str = r"F:\code\server_models";
// WSL 发行版 / conda 环境 / 启动命令统一由项目根下的 start_backend.bat、stop_backend.bat 维护

/// 追加一行到日志文件；失败静默，日志本身不能影响主流程
fn append_log(path: &std::path::Path, line: &str) {
    use std::io::Write;
    if let Ok(mut f) = fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(f, "{line}");
    }
}

/// wsl.exe 自身的消息是 UTF-16LE，Linux 进程输出是 UTF-8，按 NUL 密度判别后统一解码
fn decode_output(bytes: &[u8]) -> String {
    let nuls = bytes.iter().filter(|b| **b == 0).count();
    if nuls * 4 > bytes.len() {
        let units: Vec<u16> = bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        return String::from_utf16_lossy(&units);
    }
    String::from_utf8_lossy(bytes).into_owned()
}

/// 原样追加文本（不加换行），用于转发子进程输出
fn append_raw(path: &std::path::Path, text: &str) {
    use std::io::Write;
    if let Ok(mut f) = fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = f.write_all(text.as_bytes());
    }
}

/// 后台把子进程输出写入日志；按块解码直写，避免按行切分打乱 UTF-16 的 0A 00 换行
fn pipe_to_log<R: std::io::Read + Send + 'static>(mut reader: R, path: PathBuf) {
    std::thread::spawn(move || {
        let mut chunk = [0u8; 8192];
        loop {
            let n = match reader.read(&mut chunk) {
                Ok(0) | Err(_) => break,
                Ok(n) => n,
            };
            append_raw(&path, &decode_output(&chunk[..n]));
        }
    });
}

/// 目录存在不代表可写，这里实际写探针文件确认
#[cfg(windows)]
fn is_writable(dir: &std::path::Path) -> bool {
    if fs::create_dir_all(dir).is_err() {
        return false;
    }
    let probe = dir.join(".write-probe");
    let ok = fs::write(&probe, b"x").is_ok();
    let _ = fs::remove_file(&probe);
    ok
}

/// 优先使用传入目录；不可写时退回 exe 同级目录，保证 WebView2 与日志总能落地
#[cfg(windows)]
fn writable_dir(preferred: PathBuf) -> PathBuf {
    let fallback = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(PathBuf::from))
        .unwrap_or_else(std::env::temp_dir);

    let candidates = [preferred, fallback.clone()];
    candidates
        .into_iter()
        .find(|dir| is_writable(dir))
        .unwrap_or(fallback)
}

/// WebView2 用户数据目录：优先放项目内，避开 %LOCALAPPDATA% 不可写或被占用导致窗口创建失败
#[cfg(windows)]
fn webview_data_dir() -> PathBuf {
    writable_dir(
        PathBuf::from(PROJECT_WIN_PATH)
            .join("assets")
            .join("runtime")
            .join("webview"),
    )
}

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
    let bat = PathBuf::from(PROJECT_WIN_PATH).join("start_backend.bat");
    if !bat.is_file() {
        return Err(format!("缺少启动脚本: {}", bat.display()));
    }

    let log_path = writable_dir(PathBuf::from(PROJECT_WIN_PATH).join("assets").join("runtime"))
        .join("backend.log");

    // 8666 已有服务时直接复用：重复拉起只会让新 python 抢端口失败
    if backend_ready() {
        append_log(&log_path, "===== 8666 已在监听，复用现有后端 =====");
        return Ok("后端已在运行（8666 正在监听），跳过重复拉起".into());
    }

    append_log(&log_path, &format!("\n===== 调用 {} =====", bat.display()));

    // 脚本只负责拉起，stdout/stderr 全部由这里接管写盘，方便运行期间随时读取
    let mut child = run_bat(&bat)?;
    if let Some(out) = child.stdout.take() {
        pipe_to_log(out, log_path.clone());
    }
    if let Some(err) = child.stderr.take() {
        pipe_to_log(err, log_path.clone());
    }
    let wait_log = log_path.clone();
    std::thread::spawn(move || match child.wait() {
        Ok(status) => append_log(&wait_log, &format!("===== 启动脚本退出码 {:?} =====", status.code())),
        Err(e) => append_log(&wait_log, &format!("===== 等待启动脚本失败: {e} =====")),
    });

    Ok(format!("已执行 {}，日志 {}", bat.display(), log_path.display()))
}

/// 网关端口是否已经有服务在监听
#[cfg(windows)]
fn backend_ready() -> bool {
    use std::net::{SocketAddr, TcpStream};
    use std::time::Duration;

    match "127.0.0.1:8666".parse::<SocketAddr>() {
        Ok(addr) => TcpStream::connect_timeout(&addr, Duration::from_millis(300)).is_ok(),
        Err(_) => false,
    }
}

/// 执行项目内的 .bat：隐藏窗口，stdin 给 null，输出交由调用方接管
#[cfg(windows)]
fn run_bat(bat: &std::path::Path) -> Result<std::process::Child, String> {
    // 必须带上 CREATE_UNICODE_ENVIRONMENT，否则 creation_flags 会覆盖 Rust 默认值，环境块编码错乱
    Command::new("cmd.exe")
        .arg("/c")
        .arg(bat)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .creation_flags(CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT)
        .spawn()
        .map_err(|e| format!("执行 {} 失败: {}", bat.display(), e))
}

#[cfg(not(windows))]
#[tauri::command]
fn start_wsl_backend() -> Result<String, String> {
    Err("仅 Windows 支持拉起 WSL 后端，当前平台将直接连接网关".into())
}

/// 前端握手失败时读取后端日志尾部，直接在界面显示原因
#[cfg(windows)]
#[tauri::command]
fn read_backend_log() -> Result<String, String> {
    let path = writable_dir(PathBuf::from(PROJECT_WIN_PATH).join("assets").join("runtime"))
        .join("backend.log");
    // 日志里混有 WSL 的 UTF-8、wsl.exe 的 UTF-16、cmd 的 GBK，不能按 UTF-8 严格解码
    let bytes = fs::read(&path).map_err(|e| format!("读取 {} 失败: {}", path.display(), e))?;
    let text = String::from_utf8_lossy(&bytes);
    let lines: Vec<&str> = text.lines().collect();
    let start = lines.len().saturating_sub(40);
    Ok(lines[start..].join("\n"))
}

#[cfg(not(windows))]
#[tauri::command]
fn read_backend_log() -> Result<String, String> {
    Err("仅 Windows 支持读取后端日志".into())
}

#[cfg(windows)]
#[tauri::command]
fn stop_wsl_backend() -> Result<String, String> {
    let bat = PathBuf::from(PROJECT_WIN_PATH).join("stop_backend.bat");
    if !bat.is_file() {
        return Err(format!("缺少关闭脚本: {}", bat.display()));
    }
    let out = run_bat(&bat)?
        .wait_with_output()
        .map_err(|e| format!("等待 {} 结束失败: {}", bat.display(), e))?;
    Ok(format!("已执行 {}，退出码 {:?}", bat.display(), out.status.code()))
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

/// release 构建无控制台，启动失败时看不到原因，这里把 panic 与启动现场落盘便于排查
#[cfg(windows)]
fn install_crash_logger() {
    use std::io::Write;

    fn describe(result: &std::io::Result<()>) -> String {
        match result {
            Ok(()) => "成功".into(),
            Err(e) => format!("失败 {e}"),
        }
    }

    let dir = writable_dir(PathBuf::from(PROJECT_WIN_PATH).join("assets").join("runtime"));
    let log = dir.join("client-error.log");

    let data_dir = webview_data_dir();
    let probe = data_dir.join(".write-probe");
    let mk = fs::create_dir_all(&data_dir);
    let wr = fs::write(&probe, b"x");
    let _ = fs::remove_file(&probe);

    let _ = fs::write(
        &log,
        format!(
            "启动 exe={}\nWebView2 数据目录={}\n  建目录={}\n  写文件={}\n",
            std::env::current_exe().map(|p| p.display().to_string()).unwrap_or_default(),
            data_dir.display(),
            describe(&mk),
            describe(&wr),
        ),
    );

    std::panic::set_hook(Box::new(move |info| {
        if let Ok(mut f) = fs::OpenOptions::new().append(true).create(true).open(&log) {
            let _ = writeln!(f, "\n[崩溃] {info}");
        }
    }));
}

/// 按 tauri.conf.json 的窗口配置建主窗口，Windows 下强制指定 WebView2 数据目录
fn build_main_window(app: &mut tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    let config = app
        .config()
        .app
        .windows
        .iter()
        .find(|w| w.label == "main")
        .cloned()
        .ok_or("tauri.conf.json 中缺少 label=main 的窗口配置")?;

    let mut builder = WebviewWindowBuilder::from_config(app.handle(), &config)?;

    #[cfg(windows)]
    {
        let data_dir = webview_data_dir();
        fs::create_dir_all(&data_dir)?;
        builder = builder.data_directory(data_dir);
    }

    builder.build()?;
    Ok(())
}

fn main() {
    #[cfg(windows)]
    install_crash_logger();

    tauri::Builder::default()
        .setup(|app| {
            build_main_window(app)?;
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            start_wsl_backend,
            stop_wsl_backend,
            read_json_config,
            write_json_config,
            read_backend_log
        ])
        .run(tauri::generate_context!())
        .expect("Tauri 运行异常");
}
