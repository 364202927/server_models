import { useEffect } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { useAppStore } from "./store/useAppStore";
import { desktopInvoke, isTauri } from "./lib/bridge";
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
        await desktopInvoke("start_wsl_backend");
      } catch (err) {
        appendLog("WARN", `未拉起本地后端，尝试直连网关: ${err}`);
      }

      // 握手探测 (最多等待 15 次，每次间隔 1s)
      let connected = false;
      for (let i = 0; i < 15; i++) {
        if (isCancelled) return;
        await new Promise((r) => setTimeout(r, 1000));
        const pong = await sendAdminCmd(1000);
        if (pong?.status === "ready") {
          connected = true;
          break;
        }
      }

      if (!connected || isCancelled) {
        useAppStore.setState({ engineState: "error" });
        return;
      }

      await loadRemoteModels();

      // 界面完全就绪后，唯一一次自动触发后端加载模型
      useAppStore.setState({ engineState: "launching" });
      await sendAdminCmd(1001);

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
          sendAdminCmd(1099),
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
