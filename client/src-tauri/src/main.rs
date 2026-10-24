#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::fs;
use std::path::PathBuf;

#[cfg(windows)]
use std::os::windows::process::CommandExt;
#[cfg(windows)]
use std::process::{Command, Stdio};
use tauri::menu::{Menu, MenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{Manager, WebviewWindowBuilder};

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
// 调用 start/stop_backend.bat 的口令，脚本里校验同一个值；双击脚本没有参数，会直接退出
#[cfg(windows)]
const BACKEND_TOKEN: &str = "SM-6f1a9c3e8b2d4705";
// WSL 发行版 / conda 环境 / 启动命令统一由 client 目录下的 start_backend.bat、stop_backend.bat 维护

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

/// 启动/关闭脚本放在 client 目录，脚本内校验口令，只有桌面端能调用（双击无效）
#[cfg(windows)]
fn backend_bat(name: &str) -> PathBuf {
    PathBuf::from(PROJECT_WIN_PATH).join("client").join(name)
}

#[cfg(windows)]
#[tauri::command]
fn start_wsl_backend() -> Result<String, String> {
    let bat = backend_bat("start_backend.bat");
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

/// 执行 client 下的 .bat：隐藏窗口，stdin 给 null，输出交由调用方接管，首个参数是脚本要求的调用口令
#[cfg(windows)]
fn run_bat(bat: &std::path::Path) -> Result<std::process::Child, String> {
    // 必须带上 CREATE_UNICODE_ENVIRONMENT，否则 creation_flags 会覆盖 Rust 默认值，环境块编码错乱
    Command::new("cmd.exe")
        .arg("/c")
        .arg(bat)
        .arg(BACKEND_TOKEN)
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
    let bat = backend_bat("stop_backend.bat");
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
    let text = fs::read_to_string(&path).map_err(|e| format!("读取文件失败 {:?}: {}", path, e))?;
    // 编辑器另存为 UTF-8 with BOM 时会在开头留 U+FEFF，前端 JSON.parse 会直接报错
    Ok(text.trim_start_matches('\u{feff}').to_string())
}

#[tauri::command]
fn write_json_config(filename: String, content: String) -> Result<(), String> {
    let target = config_path(&filename)?;
    let temp = target.with_extension("json.tmp");

    // 原子写入：先写临时文件再覆盖，防止写坏 JSON
    fs::write(&temp, content).map_err(|e| format!("写入临时文件失败: {}", e))?;
    fs::rename(&temp, &target).map_err(|e| format!("覆盖配置文件失败: {}", e))
}

/// 目录选择：复用系统 FolderBrowserDialog，避免为单个功能引入 dialog 插件依赖
/// 选中结果写入临时文件再读回，规避控制台编码对中文路径的影响；取消选择返回 None
#[tauri::command]
async fn pick_directory() -> Result<Option<String>, String> {
    #[cfg(not(windows))]
    return Ok(None);

    #[cfg(windows)]
    {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let out_file = std::env::temp_dir().join(format!("sm-pick-{}-{}.txt", std::process::id(), stamp));
        // 单引号字符串里的 ' 需要写成 ''，防止临时路径带引号时脚本被截断
        let target = out_file.display().to_string().replace('\'', "''");
        let script = format!(
            "Add-Type -AssemblyName System.Windows.Forms\n\
             $dialog = New-Object System.Windows.Forms.FolderBrowserDialog\n\
             $dialog.Description = '选择模型路径'\n\
             if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {{\n\
                [System.IO.File]::WriteAllText('{target}', $dialog.SelectedPath, [System.Text.UTF8Encoding]::new($false))\n\
             }}"
        );

        let output = tauri::async_runtime::spawn_blocking(move || {
            Command::new("powershell.exe")
                .args(["-NoProfile", "-STA", "-NonInteractive", "-Command", &script])
                .creation_flags(CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT)
                .output()
        })
        .await
        .map_err(|e| format!("目录选择任务异常: {e}"))?
        .map_err(|e| format!("启动目录选择失败: {e}"))?;

        let selected = fs::read_to_string(&out_file)
            .ok()
            .map(|text| text.trim().to_string())
            .filter(|text| !text.is_empty());
        let _ = fs::remove_file(&out_file);

        if !output.status.success() {
            return Err(format!("目录选择失败: {}", decode_output(&output.stderr)));
        }
        Ok(selected)
    }
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

/// 从托盘恢复主窗口：必须先解除最小化，否则窗口显示出来仍是最小化状态
fn show_main_window(app: &tauri::AppHandle) {
    let Some(window) = app.get_webview_window("main") else { return };
    let _ = window.unminimize();
    let _ = window.show();
    let _ = window.set_focus();
}

/// 托盘“退出”复用窗口关闭流程，前端 onCloseRequested 负责保存日志、释放显存、停后端
fn request_quit(app: &tauri::AppHandle) {
    let Some(window) = app.get_webview_window("main") else {
        app.exit(0);
        return;
    };
    let _ = window.close();
}

/// 托盘图标：左键单击恢复窗口，右键菜单提供“显示主界面 / 退出”
fn setup_tray(app: &mut tauri::App) -> tauri::Result<()> {
    let show = MenuItem::with_id(app, "show", "显示主界面", true, None::<&str>)?;
    let quit = MenuItem::with_id(app, "quit", "退出", true, None::<&str>)?;
    let menu = Menu::with_items(app, &[&show, &quit])?;

    let mut tray = TrayIconBuilder::with_id("main-tray")
        .tooltip("SERVER MODELS")
        .menu(&menu)
        .show_menu_on_left_click(false)
        .on_menu_event(|app, event| match event.id.as_ref() {
            "show" => show_main_window(app),
            "quit" => request_quit(app),
            _ => {}
        })
        .on_tray_icon_event(|tray, event| {
            // 只响应左键抬起，避免右键菜单与单击恢复互相干扰
            let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            else {
                return;
            };
            show_main_window(tray.app_handle());
        });

    // 复用 exe 内嵌的应用图标；取不到时仍建托盘，至少右键菜单可用
    if let Some(icon) = app.default_window_icon() {
        tray = tray.icon(icon.clone());
    }

    tray.build(app)?;
    Ok(())
}

fn main() {
    #[cfg(windows)]
    install_crash_logger();

    tauri::Builder::default()
        // 单实例守卫必须最先注册：重复启动时新进程只把已有窗口提到前台，随后自行退出
        .plugin(tauri_plugin_single_instance::init(|app, _argv, _cwd| {
            show_main_window(app);
        }))
        // 最小化即隐藏到托盘：Windows 最小化会触发 Resized，此时窗口已是最小化状态
        .on_window_event(|window, event| {
            if window.label() != "main" {
                return;
            }
            if let tauri::WindowEvent::Resized(_) = event {
                if window.is_minimized().unwrap_or(false) {
                    let _ = window.hide();
                }
            }
        })
        .setup(|app| {
            build_main_window(app)?;
            setup_tray(app)?;
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            start_wsl_backend,
            stop_wsl_backend,
            read_json_config,
            write_json_config,
            read_backend_log,
            pick_directory
        ])
        .run(tauri::generate_context!())
        .expect("Tauri 运行异常");
}
