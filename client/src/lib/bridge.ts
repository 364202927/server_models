import { invoke } from "@tauri-apps/api/core";

/** 是否运行在 Tauri 桌面壳内（Windows 桌面端）；Mac 浏览器预览时为 false */
export const isTauri = (): boolean => typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

/** 调用桌面壳命令；非桌面环境直接拒绝，由调用方决定如何降级 */
export const desktopInvoke = <T = void>(cmd: string, args?: Record<string, unknown>): Promise<T> => {
  if (!isTauri()) return Promise.reject(new Error("当前不是桌面端环境"));
  return invoke<T>(cmd, args);
};
