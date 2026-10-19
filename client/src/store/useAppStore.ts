import { create } from "zustand";
import { desktopInvoke, isTauri } from "../lib/bridge";
import {
  ADMIN,
  formatHardware,
  gatewayBase,
  gatewayHeaders,
  gatewayState,
  unwrapAdmin,
} from "../lib/gateway";

export type EngineState = "connecting" | "launching" | "running" | "idle" | "error";

export interface LogRecord {
  seq_id?: number;
  tags: string;
  data: { msg: string };
}

interface AppState {
  activePage: "config" | "test" | "monitor";
  engineState: EngineState;
  logs: LogRecord[];
  cursor: number;
  hardware: string;
  processes: Array<Record<string, any>>;
  settings: Record<string, any>;
  models: Record<string, any>;
  // 网关 /v1/models 返回的可用模型，不依赖本地配置文件（Mac 预览时也能拿到）
  remoteModelIds: string[];

  setActivePage: (page: "config" | "test" | "monitor") => void;
  appendLog: (tags: string, msg: string) => void;
  loadLocalConfigs: () => Promise<void>;
  loadRemoteModels: () => Promise<void>;
  saveLocalConfig: (filename: "modelSetting.json" | "models.json", data: any) => Promise<boolean>;
  sendAdminCmd: (messageId: number, args?: any) => Promise<any>;
  triggerEngineToggle: () => Promise<void>;
  pollStatusAndLogs: () => Promise<void>;
  clearLogs: () => void;
}

/** 硬件检测含 CUDA 查询，轮询降到约 6s 一次 */
const HARDWARE_EVERY = 4;
let pollTick = 0;

/** 本地模型状态转成监控页的进程行 */
const toProcessRow = (model: Record<string, any>) => ({
  model: model.id,
  state: model.state,
  vram: model.gpu_memory_mb ?? 0,
  error: model.error,
});

export const useAppStore = create<AppState>((set, get) => ({
  activePage: "config",
  engineState: "connecting",
  logs: [],
  cursor: 0,
  hardware: "等待网关同步...",
  processes: [],
  settings: {},
  models: {},
  remoteModelIds: [],

  setActivePage: (activePage) => set({ activePage }),
  clearLogs: () => set({ logs: [] }),

  appendLog: (tags, msg) =>
    set((s) => ({ logs: [...s.logs, { tags, data: { msg } }].slice(-2000) })),

  loadLocalConfigs: async () => {
    if (!isTauri()) {
      get().appendLog("INFO", "浏览器预览模式：不读取本地配置，网关地址取自 VITE_GATEWAY_URL");
      return;
    }
    try {
      const [rawSettings, rawModels] = await Promise.all([
        desktopInvoke<string>("read_json_config", { filename: "modelSetting.json" }),
        desktopInvoke<string>("read_json_config", { filename: "models.json" }),
      ]);
      set({
        settings: JSON.parse(rawSettings || "{}"),
        models: JSON.parse(rawModels || "{}"),
      });
    } catch (err) {
      get().appendLog("ERROR", `本地配置读取异常: ${err}`);
    }
  },

  loadRemoteModels: async () => {
    const { settings } = get();
    try {
      const res = await fetch(`${gatewayBase(settings)}/v1/models`, { headers: gatewayHeaders(settings) });
      if (!res.ok) return;
      const body = await res.json();
      set({ remoteModelIds: (body.data ?? []).map((m: { id: string }) => m.id) });
    } catch {
      // 网关不可达时保持空列表，由界面回退到本地配置
    }
  },

  saveLocalConfig: async (filename, data) => {
    if (!isTauri()) {
      get().appendLog("WARN", "浏览器预览模式不支持保存本地配置，请在 Windows 桌面端操作");
      return false;
    }
    try {
      const formatted = JSON.stringify(data, null, 2);
      await desktopInvoke("write_json_config", { filename, content: formatted });
      if (filename === "modelSetting.json") set({ settings: data });
      if (filename === "models.json") set({ models: data });
      return true;
    } catch (err) {
      get().appendLog("ERROR", `配置写入异常: ${err}`);
      return false;
    }
  },

  sendAdminCmd: async (messageId, args = null) => {
    const { settings } = get();
    try {
      const res = await fetch(`${gatewayBase(settings)}/v1/chat/completions`, {
        method: "POST",
        headers: gatewayHeaders(settings),
        body: JSON.stringify({ message_id: messageId, args }),
      });
      if (!res.ok) {
        gatewayState.lastError = `HTTP ${res.status}`;
        return null;
      }
      gatewayState.lastError = "";
      return unwrapAdmin(await res.json());
    } catch (err) {
      gatewayState.lastError = String(err);
      return null;
    }
  },

  triggerEngineToggle: async () => {
    const { engineState, sendAdminCmd } = get();
    if (engineState === "connecting" || engineState === "launching") return;

    if (engineState === "running") {
      set({ engineState: "launching" });
      const res = await sendAdminCmd(ADMIN.unloadAll);
      set({ engineState: res ? "idle" : "error" });
      return;
    }

    set({ engineState: "launching" });
    const res = await sendAdminCmd(ADMIN.autolaunch);
    if (!res) set({ engineState: "error" });
  },

  pollStatusAndLogs: async () => {
    const { cursor, sendAdminCmd } = get();

    // 1008：增量日志，回包为 {cursor, logs}
    const logRes = await sendAdminCmd(ADMIN.logs, { cursor });
    const newLogs: LogRecord[] = Array.isArray(logRes?.logs) ? logRes.logs : [];
    if (logRes) {
      set((state) => ({
        cursor: logRes.cursor ?? state.cursor,
        logs: newLogs.length > 0 ? [...state.logs, ...newLogs].slice(-2000) : state.logs,
      }));
    }

    // 1001：系统状态，回包为 {models, queue_length, engine_state}
    const status = await sendAdminCmd(ADMIN.status);
    if (status) {
      const models: Record<string, any>[] = Array.isArray(status.models) ? status.models : [];
      set((state) => ({
        processes: models.map(toProcessRow),
        engineState: (status.engine_state as EngineState) || state.engineState,
      }));
    }

    // 1002：硬件信息，回包是 HardwareInfo 字典，降频拉取
    if (pollTick++ % HARDWARE_EVERY !== 0) return;
    const text = formatHardware(await sendAdminCmd(ADMIN.hardware));
    if (text) set({ hardware: text });
  },
}));
