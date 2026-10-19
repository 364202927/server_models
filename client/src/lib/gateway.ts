type Settings = Record<string, any>;

/** 后端管理指令 ID，必须与 server_models/__init__.py 的 adim_ID 保持一致 */
export const ADMIN = {
  handshake: 1000,
  status: 1001,
  hardware: 1002,
  models: 1003,
  logs: 1008,
  autolaunch: 1009,
  unloadAll: 1010,
  stopped: 1099,
} as const;

/** 网关地址：优先 VITE_GATEWAY_URL（Mac 远程连接 WSL），否则连本机 modelSetting.json 中的端口 */
export const gatewayBase = (settings: Settings): string => {
  const override = (import.meta.env.VITE_GATEWAY_URL ?? "").trim();
  if (override) return override.replace(/\/+$/, "");
  const port = settings?.server?.api_port || 8666;
  return `http://127.0.0.1:${port}`;
};

/** 请求头：API Key 优先取 VITE_GATEWAY_API_KEY，其次取 modelSetting.json 中的 server.api_key */
export const gatewayHeaders = (settings: Settings): Record<string, string> => {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  const apiKey = ((import.meta.env.VITE_GATEWAY_API_KEY ?? "") || settings?.server?.api_key || "").trim();
  if (apiKey) headers["x-api-key"] = apiKey;
  return headers;
};

/** 最近一次网关请求的失败原因，握手失败时显示到界面，避免只看到"无响应" */
export const gatewayState = { lastError: "" };

/**
 * 拆解后端统一回包：{request_id, status, model, message_id, response}
 * 业务数据统一放在 response 里，且 dict 会被 json.dumps 成字符串。
 */
export const unwrapAdmin = (body: unknown): any => {
  if (!body || typeof body !== "object") return body;
  const inner = (body as Record<string, unknown>).response;
  if (typeof inner !== "string") return body;
  try {
    return JSON.parse(inner);
  } catch {
    return body;
  }
};

/** 把 1002 返回的 HardwareInfo 字典格式化成监控页展示文本 */
export const formatHardware = (info: any): string => {
  if (!info || typeof info !== "object") return "";
  const gpus: any[] = Array.isArray(info.gpus) ? info.gpus : [];
  const lines = gpus.map(
    (gpu) => `GPU${gpu.index} ${gpu.name}: 显存 ${gpu.memory_total_mb}MB, 剩余 ${gpu.memory_free_mb}MB`,
  );
  if (lines.length === 0) lines.push("未检测到 GPU");
  const memory = info.memory;
  if (memory) lines.push(`内存: 剩余 ${memory.available_mb}MB / 总计 ${memory.total_mb}MB`);
  if (info.platform) lines.push(`平台: ${info.platform}`);
  return lines.join("\n");
};
