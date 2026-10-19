import React from "react";
import { Settings, MessageSquare, Activity, Power, Loader2 } from "lucide-react";
import { useAppStore } from "../store/useAppStore";

export const Sidebar: React.FC = () => {
  const { activePage, setActivePage, engineState, triggerEngineToggle, processes } = useAppStore();

  const navItems = [
    { id: "monitor", label: "硬件与进程监控", icon: Activity },
    { id: "config", label: "LLM配置", icon: Settings },
    { id: "test", label: "模型测试", icon: MessageSquare },
  ] as const;

  // 已加载 = 运行中 + 休眠在内存；停止按钮对两者都生效
  const running = processes.filter((p) => p.state === "RUNNING").length;
  const sleeping = processes.filter((p) => p.state === "SLEEPING_RAM").length;
  const stateText = { connecting: "连接中", launching: "处理中", running: "运行中", idle: "空闲", error: "异常" }[engineState];

  const btnMeta = {
    connecting: { text: "网关启动中...", bg: "bg-slate-700 text-slate-400 cursor-not-allowed", spin: true },
    launching: { text: "模型处理中...", bg: "bg-amber-600 text-white cursor-wait", spin: true },
    running: { text: "停止模型 (释放显存)", bg: "bg-rose-600 hover:bg-rose-500 text-white", spin: false },
    idle: { text: "启动模型", bg: "bg-emerald-600 hover:bg-emerald-500 text-white", spin: false },
    error: { text: "重试启动模型", bg: "bg-red-700 hover:bg-red-600 text-white", spin: false },
  }[engineState];

  return (
    <aside className="w-56 bg-slate-900 border-r border-slate-800 flex flex-col justify-between p-3">
      <div>
        <div className="px-3 py-3 text-base font-bold tracking-wider text-slate-200 border-b border-slate-800 mb-3">
          SERVER MODELS
        </div>
        <nav className="space-y-1">
          {navItems.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setActivePage(id)}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition ${activePage === id
                  ? "bg-blue-600 text-white"
                  : "text-slate-400 hover:bg-slate-800 hover:text-slate-200"
                }`}
            >
              <Icon size={17} />
              <span>{label}</span>
            </button>
          ))}
        </nav>
      </div>

      <div className="pt-3 border-t border-slate-800">
        <div className="flex items-center justify-between text-xs text-slate-400 mb-2 px-1">
          <span>引擎状态</span>
          <span className="font-mono">{stateText}</span>
        </div>
        <div className="mb-2 px-1 text-xs text-slate-500">
          {running + sleeping === 0 ? "未加载模型" : `运行 ${running} · 休眠 ${sleeping}`}
        </div>
        <button
          onClick={triggerEngineToggle}
          disabled={engineState === "connecting" || engineState === "launching"}
          className={`w-full flex items-center justify-center gap-2 py-2.5 px-3 rounded-lg text-sm font-medium transition ${btnMeta.bg}`}
        >
          {btnMeta.spin ? <Loader2 size={16} className="animate-spin" /> : <Power size={16} />}
          <span>{btnMeta.text}</span>
        </button>
      </div>
    </aside>
  );
};