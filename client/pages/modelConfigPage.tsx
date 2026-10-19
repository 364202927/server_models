import React, { useState, useEffect } from "react";
import { Save, Plus, Trash2 } from "lucide-react";
import { useAppStore } from "../store/useAppStore";

export const modelConfigPage: React.FC = () => {
  const { settings, models, saveLocalConfig } = useAppStore();
  const [tab, setTab] = useState<"setting" | "register">("setting");
  const [settingText, setSettingText] = useState("");
  const [modelList, setModelList] = useState<Record<string, any>>({});
  const [newModelId, setNewModelId] = useState("");
  const [tip, setTip] = useState("");

  useEffect(() => {
    setSettingText(JSON.stringify(settings, null, 2));
    setModelList(models || {});
  }, [settings, models]);

  const handleSaveSetting = async () => {
    let parsed: any;
    try {
      parsed = JSON.parse(settingText);
    } catch {
      setTip("❌ JSON 格式错误，请检查语法");
      return;
    }
    const ok = await saveLocalConfig("modelSetting.json", parsed);
    setTip(ok ? "✅ modelSetting.json 已保存" : "❌ 保存失败");
  };

  const handleSaveModels = async (nextModels: Record<string, any>) => {
    setModelList(nextModels);
    const ok = await saveLocalConfig("models.json", nextModels);
    setTip(ok ? "✅ models.json 已保存（下次启动模型生效）" : "❌ 保存失败");
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <div className="flex gap-2">
          <button
            onClick={() => { setTab("setting"); setTip(""); }}
            className={`px-4 py-1.5 rounded-lg text-sm font-medium ${
              tab === "setting" ? "bg-blue-600 text-white" : "bg-slate-800 text-slate-400"
            }`}
          >
            全局配置 (modelSetting.json)
          </button>
          <button
            onClick={() => { setTab("register"); setTip(""); }}
            className={`px-4 py-1.5 rounded-lg text-sm font-medium ${
              tab === "register" ? "bg-blue-600 text-white" : "bg-slate-800 text-slate-400"
            }`}
          >
            注册模型 (models.json)
          </button>
        </div>
        {tip && <span className="text-xs text-emerald-400">{tip}</span>}
      </div>

      {tab === "setting" && (
        <div className="space-y-3">
          <textarea
            value={settingText}
            onChange={(e) => setSettingText(e.target.value)}
            className="w-full h-96 bg-slate-900 border border-slate-800 rounded-lg p-3 font-mono text-xs text-slate-200 focus:outline-none focus:border-blue-500"
          />
          <button
            onClick={handleSaveSetting}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded-lg text-sm font-medium"
          >
            <Save size={15} />
            <span>保存配置</span>
          </button>
        </div>
      )}

      {tab === "register" && (
        <div className="space-y-4">
          <div className="flex gap-2">
            <input
              value={newModelId}
              onChange={(e) => setNewModelId(e.target.value)}
              placeholder="输入新模型 ID (如 qwen2.5-7b)"
              className="bg-slate-900 border border-slate-800 rounded-lg px-3 py-1.5 text-sm w-72 focus:outline-none focus:border-blue-500"
            />
            <button
              onClick={() => {
                if (!newModelId.trim()) return;
                handleSaveModels({
                  ...modelList,
                  [newModelId.trim()]: { framework: "llama_cpp", model_path: "/mnt/d/models/xxx.gguf", launch: true },
                });
                setNewModelId("");
              }}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 rounded-lg text-sm"
            >
              <Plus size={15} />
              <span>新增模型</span>
            </button>
          </div>

          <div className="grid grid-cols-1 gap-3">
            {Object.entries(modelList).map(([id, cfg]) => (
              <div key={id} className="bg-slate-900 border border-slate-800 rounded-lg p-3 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-sm font-bold text-blue-400">{id}</span>
                  <button
                    onClick={() => {
                      const copy = { ...modelList };
                      delete copy[id];
                      handleSaveModels(copy);
                    }}
                    className="text-slate-400 hover:text-rose-400"
                  >
                    <Trash2 size={15} />
                  </button>
                </div>
                <textarea
                  defaultValue={JSON.stringify(cfg, null, 2)}
                  onBlur={(e) => {
                    try {
                      const updated = JSON.parse(e.target.value);
                      handleSaveModels({ ...modelList, [id]: updated });
                    } catch {
                      setTip(`❌ 模型 [${id}] JSON 语法有误，未保存`);
                    }
                  }}
                  className="w-full h-28 bg-slate-950 border border-slate-800 rounded p-2 font-mono text-xs text-slate-300"
                />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};