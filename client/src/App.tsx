import { useEffect } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { useAppStore } from "./store/useAppStore";
import { desktopInvoke, isTauri } from "./lib/bridge";
import { ADMIN, gatewayState } from "./lib/gateway";
import { Sidebar } from "./components/sidebar";
import { LogPanel } from "./components/logPanel";
import { ModelConfigPage } from "./pages/modelConfigPage";
import { ModelTestPage } from "./pages/modelTestPage";
import { MonitorPage } from "./pages/monitorPage";

export default function App() {
  const { activePage, loadLocalConfigs, loadRemoteModels, sendAdminCmd, pollStatusAndLogs, appendLog } = useAppStore();

  useEffect(() => {
    let pollTimer: number | undefined;
    let isCancelled = false;

    const bootSequence = async () => {
      await loadLocalConfigs();
      // 拉起失败（Mac / 浏览器预览 / 后端已在别处运行）不中断，仍尝试直连网关
      try {
        const res = await desktopInvoke<string>("start_wsl_backend");
        appendLog("INFO", `已触发后端拉起: ${res}`);
      } catch (err) {
        appendLog("WARN", `未拉起本地后端，尝试直连网关: ${err}`);
      }

      // 握手探测：WSL 冷启动 + conda 激活 + python 加载 torch 通常要 20s 以上，给到 60s
      let connected = false;
      for (let i = 0; i < 60; i++) {
        if (isCancelled) return;
        await new Promise((r) => setTimeout(r, 1000));
        // 回包已在 sendAdminCmd 里拆掉了外层信封，业务状态直接取 status
        const pong = await sendAdminCmd(ADMIN.handshake);
        if (pong?.status === "ready") {
          connected = true;
          appendLog("INFO", `网关握手成功（第 ${i + 1} 次，耗时约 ${i + 1}s）`);
          break;
        }
        if (i > 0 && (i + 1) % 10 === 0) appendLog("INFO", `等待网关握手... 已重试 ${i + 1} 次`);
      }

      if (!connected || isCancelled) {
        const reason = gatewayState.lastError || "无响应";
        appendLog("ERROR", `网关握手失败：8666 端口 60s 内无响应（最近错误: ${reason}）`);
        if (isTauri()) {
          try {
            const tail = await desktopInvoke<string>("read_backend_log");
            tail.split("\n").filter(Boolean).forEach((line) => appendLog("BACKEND", line));
          } catch (err) {
            appendLog("WARN", `读取后端日志失败: ${err}`);
          }
        }
        useAppStore.setState({ engineState: "error" });
        return;
      }

      await loadRemoteModels();
      const modelCount = useAppStore.getState().remoteModelIds.length;
      appendLog("INFO", `网关返回可用模型 ${modelCount} 个`);

      // 后端 autolaunch 会自行加载 launch=true 的模型，这里只拉一次状态并开始轮询
      useAppStore.setState({ engineState: "launching" });
      await sendAdminCmd(ADMIN.status);
      appendLog("INFO", "已请求后端状态，等待模型加载（关注下方轮询日志）");

      // 开启 1.5s 状态与日志低频轮询
      pollTimer = window.setInterval(() => {
        pollStatusAndLogs();
      }, 1500);
    };

    bootSequence();

    // 仅桌面端拦截关闭：先发 1099 保存日志释放显存，再关停 WSL
    let unlistenPromise: Promise<() => void> | undefined;
    if (isTauri()) {
      const appWindow = getCurrentWindow();
      unlistenPromise = appWindow.onCloseRequested(async (event) => {
        event.preventDefault();
        if (pollTimer) clearInterval(pollTimer);

        await Promise.race([
          sendAdminCmd(ADMIN.stopped),
          new Promise((r) => setTimeout(r, 2000)),
        ]);

        try {
          await desktopInvoke("stop_wsl_backend");
        } catch (err) {
          console.warn("关停 WSL 失败:", err);
        }
        await appWindow.destroy();
      });
    }

    return () => {
      isCancelled = true;
      if (pollTimer) clearInterval(pollTimer);
      unlistenPromise?.then((unlisten) => unlisten());
    };
  }, []);

  return (
    <div className="flex h-screen w-screen bg-slate-950 text-slate-100 overflow-hidden select-none">
      <Sidebar />
      <div className="flex flex-col flex-1 h-full overflow-hidden">
        <main className="flex-1 overflow-y-auto p-5">
          {activePage === "config" && <ModelConfigPage />}
          {activePage === "test" && <ModelTestPage />}
          {activePage === "monitor" && <MonitorPage />}
        </main>
        <LogPanel />
      </div>
    </div>
  );
}
