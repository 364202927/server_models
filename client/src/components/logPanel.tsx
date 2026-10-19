import React, { useState, useRef, useEffect } from "react";
import { ChevronDown, ChevronUp, Trash2, Terminal } from "lucide-react";
import { useAppStore } from "../store/useAppStore";

const TAG_COLORS: Record<string, string> = {
  ERROR: "text-red-400",
  WARN: "text-yellow-300",
  INFO: "text-cyan-300",
  LOG: "text-slate-200",
};

export const LogPanel: React.FC = () => {
  const { logs, clearLogs } = useAppStore();
  const [visible, setVisible] = useState(true);
  const [filter, setFilter] = useState<string>("ALL");
  const scrollRef = useRef<HTMLDivElement>(null);
  const autoScrollRef = useRef(true);

  const filteredLogs = filter === "ALL" ? logs : logs.filter((l) => String(l.tags).toUpperCase() === filter);

  useEffect(() => {
    if (visible && autoScrollRef.current && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [filteredLogs, visible]);

  return (
    <section className="border-t border-slate-800 bg-slate-900 flex flex-col">
      {/* 顶部控制条（固定 36px，点击可折叠隐藏） */}
      <div className="h-9 px-4 flex items-center justify-between bg-slate-900/90 border-b border-slate-800/80 text-xs">
        <div className="flex items-center gap-2 text-slate-300 font-medium">
          <Terminal size={14} className="text-blue-400" />
          <span>Console 日志监控 ({filteredLogs.length}/2000)</span>
        </div>

        <div className="flex items-center gap-2">
          {(["ALL", "INFO", "WARN", "ERROR"] as const).map((tag) => (
            <button
              key={tag}
              onClick={() => setFilter(tag)}
              className={`px-2 py-0.5 rounded transition ${filter === tag ? "bg-blue-600 text-white" : "bg-slate-800 text-slate-400 hover:text-slate-200"
                }`}
            >
              {tag}
            </button>
          ))}
          <button
            onClick={clearLogs}
            className="p-1 text-slate-400 hover:text-rose-400 transition"
            title="清空日志"
          >
            <Trash2 size={14} />
          </button>
          <button
            onClick={() => setVisible(!visible)}
            className="flex items-center gap-1 px-2 py-0.5 bg-slate-800 hover:bg-slate-700 rounded text-slate-300"
          >
            {visible ? <ChevronDown size={14} /> : <ChevronUp size={14} />}
            <span>{visible ? "隐藏" : "显示"}</span>
          </button>
        </div>
      </div>

      {/* 日志内容区：纯展示无输入，CSS content-visibility 保障 2000 条不卡顿 */}
      {visible && (
        <div
          ref={scrollRef}
          onScroll={(e) => {
            const el = e.currentTarget;
            autoScrollRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 25;
          }}
          className="h-60 overflow-y-auto px-4 py-2 font-mono text-xs space-y-1 bg-black/60 select-text"
        >
          {filteredLogs.length === 0 && <div className="text-slate-600 italic">暂无日志...</div>}
          {filteredLogs.map((item, idx) => {
            const tag = String(item.tags || "LOG").toUpperCase();
            const colorClass = TAG_COLORS[tag] || "text-slate-200";
            return (
              <div
                key={item.seq_id != null ? `seq-${item.seq_id}` : `local-${idx}`}
                style={{ contentVisibility: "auto", containIntrinsicSize: "20px" }}
                className={`leading-relaxed whitespace-pre-wrap break-all ${colorClass}`}
              >
                <span className="opacity-60 mr-2">[{tag}]</span>
                {item.data?.msg}
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
};