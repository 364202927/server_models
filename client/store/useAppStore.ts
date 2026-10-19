import { create } from "zustand";
import { invoke } from "@tauri-apps/api/core";

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

  setActivePage: (page: "config" | "test" | "monitor") => void;
  loadLocalConfigs: () => Promise<void>;
  saveLocalConfig: (filename: "modelSetting.json" | "models.json", data: any) => Promise<boolean>;
  sendAdminCmd: (messageId: number, args?: any) => Promise<any>;
  triggerEngineToggle: () => Promise<void>;
  pollStatusAndLogs: () => Promise<void>;
  clearLogs: () => void;
}

export const useAppStore = create<AppState>((set, get) => ({
  activePage: "config",
  engineState: "connecting",
  logs: [],
  cursor: 0,
  hardware: "等待网关同步...",
  processes: [],
  settings: {},
  models: {},

  setActivePage: (activePage) => set({ activePage }),
  clearLogs: () => set({ logs: [] }),

  loadLocalConfigs: async () => {
    try {
      const [rawSettings, rawModels] = await Promise.all([
        invoke<string>("read_json_config", { filename: "modelSetting.json" }),
        invoke<string>("read_json_config", { filename: "models.json" }),
      ]);
      set({
        settings: JSON.parse(rawSettings || "{}"),
        models: JSON.parse(rawModels || "{}"),
      });
    } catch (err) {
      set((s) => ({
        logs: [...s.logs, { tags: "ERROR", data: { msg: `本地配置读取异常: ${err}` } }].slice(-2000),
      }));
    }
  },

  saveLocalConfig: async (filename, data) => {
    try {
      const formatted = JSON.stringify(data, null, 2);
      await invoke("write_json_config", { filename, content: formatted });
      if (filename === "modelSetting.json") set({ settings: data });
      if (filename === "models.json") set({ models: data });
      return true;
    } catch (err) {
      set((s) => ({
        logs: [...s.logs, { tags: "ERROR", data: { msg: `配置写入异常: ${err}` } }].slice(-2000),
      }));
      return false;
    }
  },

  sendAdminCmd: async (messageId, args = null) => {
    const serverCfg = get().settings?.server || {};
    const port = serverCfg.api_port || 8666;
    const apiKey = (serverCfg.api_key || "").trim();

    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (apiKey) headers["x-api-key"] = apiKey;

    try {
      const res = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
        method: "POST",
        headers,
        body: JSON.stringify({ message_id: messageId, args }),
      });
      if (!res.ok) return null;
      return await res.json();
    } catch {
      return null;
    }
  },

  triggerEngineToggle: async () => {
    const { engineState, sendAdminCmd } = get();
    if (engineState === "connecting" || engineState === "launching") return;

    if (engineState === "running") {
      set({ engineState: "launching" });
      const res = await sendAdminCmd(1002);
      set({ engineState: res ? "idle" : "error" });
      return;
    }

    set({ engineState: "launching" });
    const res = await sendAdminCmd(1001);
    if (!res) set({ engineState: "error" });
  },

  pollStatusAndLogs: async () => {
    const { cursor, sendAdminCmd } = get();
    const res = await sendAdminCmd(1003, { cursor });
    if (!res) return;

    const newLogs: LogRecord[] = Array.isArray(res.logs) ? res.logs : [];
    set((state) => ({
      cursor: res.cursor ?? state.cursor,
      logs: newLogs.length > 0 ? [...state.logs, ...newLogs].slice(-2000) : state.logs,
      hardware: res.hardware || state.hardware,
      processes: res.processes || state.processes,
      engineState: res.is_running ? "running" : state.engineState === "launching" ? "launching" : "idle",
    }));
  },
}));