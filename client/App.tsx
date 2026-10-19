import React, { useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { useAppStore } from "./store/useAppStore";
import { Sidebar } from "./components/Sidebar";
import { LogPanel } from "./components/LogPanel";
import { ModelConfigPage } from "./pages/modelConfigPage.tsx";
import { ModelTestPage } from "./pages/ModelTestPage";
import { MonitorPage } from "./pages/MonitorPage";

export default function App() {
  const { activePage, loadLocalConfigs, sendAdminCmd, pollStatusAndLogs } = useAppStore();

  useEffect(() => {
    let pollTimer: number | undefined;
    let isCancelled = false;

    const bootSequence = async () => {
      await loadLocalConfigs();
      try {
        await invoke("start_wsl_backend");
      } catch (err) {
        useAppStore.setState({ engineState: "error" });
        return;
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

      // 界面完全就绪后，唯一一次自动触发后端加载模型
      useAppStore.setState({ engineState: "launching" });
      await sendAdminCmd(1001);

      // 开启 1.5s 状态与日志低频轮询
      pollTimer = window.setInterval(() => {
        pollStatusAndLogs();
      }, 1500);
    };

    bootSequence();

    // 拦截窗口右上角关闭：先发 1099 保存日志释放显存，再关停 WSL
    const appWindow = getCurrentWindow();
    const unlistenPromise = appWindow.onCloseRequested(async (event) => {
      event.preventDefault();
      if (pollTimer) clearInterval(pollTimer);

      await Promise.race([
        sendAdminCmd(1099),
        new Promise((r) => setTimeout(r, 2000)),
      ]);

      try {
        await invoke("stop_wsl_backend");
      } finally {
        await appWindow.destroy();
      }
    });

    return () => {
      isCancelled = true;
      if (pollTimer) clearInterval(pollTimer);
      unlistenPromise.then((unlisten) => unlisten());
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