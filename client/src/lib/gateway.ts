type Settings = Record<string, any>;

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
