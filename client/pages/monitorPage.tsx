import React from "react";
import { Cpu, Server } from "lucide-react";
import { useAppStore } from "../store/useAppStore";

export const monitorPage: React.FC = () => {
  const { hardware, processes } = useAppStore();

  return (
    <div className="space-y-5">
      <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 space-y-2">
        <div className="flex items-center gap-2 text-sm font-semibold text-blue-400">
          <Cpu size={16} />
          <span>硬件实时状态 (hardware_summary)</span>
        </div>
        <pre className="text-xs font-mono text-slate-300 bg-slate-950 p-3 rounded border border-slate-800 whitespace-pre-wrap select-text">
          {hardware}
        </pre>
      </div>

      <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 space-y-3">
        <div className="flex items-center gap-2 text-sm font-semibold text-emerald-400">
          <Server size={16} />
          <span>活跃模型进程列表</span>
        </div>
        {processes.length === 0 ? (
          <div className="text-xs text-slate-500 italic">当前无驻留模型子进程</div>
        ) : (
          <div className="grid grid-cols-2 gap-3">
            {processes.map((proc, idx) => (
              <div key={idx} className="bg-slate-950 border border-slate-800 rounded p-3 text-xs font-mono space-y-1">
                <div className="text-slate-200 font-bold">Model: {proc.model || "-"}</div>
                <div className="text-slate-400">PID: {proc.pid || "-"}</div>
                <div className="text-slate-400">空闲回收倒计时: {proc.idle_left ?? "-"}s</div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};