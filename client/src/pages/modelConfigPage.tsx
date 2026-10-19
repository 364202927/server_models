import React, { useEffect, useState } from "react";
import { Save, Plus, Trash2 } from "lucide-react";
import { useAppStore } from "../store/useAppStore";
import { ConfigFields, FieldGroup, ValueField, isPlainObject, type Json } from "../components/configFields";
import { ENGINE_OPTIONS, FIELD_OPTIONS } from "../lib/configOptions";

type Tab = "models" | "settings";

const TAB_LABELS: Record<Tab, string> = { models: "配置", settings: "设定" };
const FILE_NAMES: Record<Tab, string> = { models: "models.json", settings: "modelSetting.json" };

/** 菜单项：key 对应数据键，alert 为红字高亮（launch 为 true 的模型） */
interface MenuItem {
  key: string;
  alert?: boolean;
}

/** 首项“默认设置”在 models.json 对应 defaults，在 modelSetting.json 对应 template */
const DEFAULT_KEY: Record<Tab, string> = { models: "defaults", settings: "template" };

const copyOf = (v: unknown): Json => JSON.parse(JSON.stringify(v ?? {}));

/** 新增项的初始数据：load 与 generation 均从 modelSetting.json 的 template 拷贝 */
const fromTemplate = (template: Json): Json => ({
  load: copyOf(template.load),
  generation: copyOf(template.generation),
});

const newModel = (template: Json): Json => ({
  path: "",
  engine: "llama",
  launch: false,
  data_compression: "default",
  ...fromTemplate(template),
});

/** modelSetting.json 中的引擎专属设定（template 与下划线注释除外） */
const engineKeysOf = (settings: Json): string[] =>
  Object.keys(settings).filter((k) => k !== "template" && !k.startsWith("_") && isPlainObject(settings[k]));

export const ModelConfigPage: React.FC = () => {
  const { settings, models, saveLocalConfig } = useAppStore();
  const [tab, setTab] = useState<Tab>("models");
  const [selected, setSelected] = useState("");
  const [draftModels, setDraftModels] = useState<Json>({});
  const [draftSettings, setDraftSettings] = useState<Json>({});
  const [newName, setNewName] = useState("");
  const [tip, setTip] = useState("");

  // 两份草稿分别同步：保存其中一个文件时，不覆盖另一个尚未保存的修改
  useEffect(() => {
    setDraftSettings(settings || {});
  }, [settings]);
  useEffect(() => {
    setDraftModels(models || {});
  }, [models]);

  const template: Json = draftSettings.template ?? {};
  // models.json 的模型位于 models 键下，defaults / launch / group 与之并列
  const modelMap: Json = draftModels.models ?? {};
  const modelIds = Object.keys(modelMap);
  const engines = engineKeysOf(draftSettings);

  // 左侧菜单：配置页为 defaults + models.json 中的模型（launch 为 true 红字），设定页为 template + 引擎
  const menu: MenuItem[] = tab === "models"
    ? [{ key: "defaults" }, ...modelIds.map((id) => ({ key: id, alert: modelMap[id]?.launch === true }))]
    : [{ key: "template" }, ...engines.map((name) => ({ key: name }))];
  const activeKey = menu.some((i) => i.key === selected) ? selected : menu[0].key;
  const isDefault = activeKey === DEFAULT_KEY[tab];

  // 中间与右侧共用的当前数据：配置页 defaults 来自 models.json 顶层，其余模型来自 models 键
  const item: Json = (tab === "models"
    ? (activeKey === "defaults" ? draftModels.defaults : modelMap[activeKey])
    : draftSettings[activeKey]) ?? {};

  const setItem = (next: Json) => {
    if (tab === "settings") {
      setDraftSettings({ ...draftSettings, [activeKey]: next });
    } else if (activeKey === "defaults") {
      setDraftModels({ ...draftModels, defaults: next });
    } else {
      setDraftModels({ ...draftModels, models: { ...modelMap, [activeKey]: next } });
    }
  };

  const dirty = tab === "models"
    ? JSON.stringify(draftModels) !== JSON.stringify(models || {})
    : JSON.stringify(draftSettings) !== JSON.stringify(settings || {});

  const handleSave = async () => {
    const ok = tab === "models"
      ? await saveLocalConfig("models.json", draftModels)
      : await saveLocalConfig("modelSetting.json", draftSettings);
    setTip(ok ? `✅ ${FILE_NAMES[tab]} 已保存（下次启动生效）` : "❌ 保存失败");
  };

  const addItem = () => {
    const name = newName.trim();
    if (!name) return;
    if (tab === "models") {
      if (name in modelMap || name === "defaults") {
        setTip(`❌ 模型 [${name}] 已存在`);
        return;
      }
      setDraftModels({ ...draftModels, models: { ...modelMap, [name]: newModel(template) } });
      setSelected(name);
    } else {
      const engine = name.toLowerCase();
      if (engine === "template" || engine in draftSettings) {
        setTip(`❌ 引擎 [${engine}] 无效或已存在`);
        return;
      }
      setDraftSettings({ ...draftSettings, [engine]: fromTemplate(template) });
      setSelected(engine);
    }
    setNewName("");
  };

  const removeItem = () => {
    if (tab === "models") {
      const next = { ...modelMap };
      delete next[activeKey];
      setDraftModels({ ...draftModels, models: next });
    } else {
      const next = { ...draftSettings };
      delete next[activeKey];
      setDraftSettings(next);
    }
    setSelected("");
  };

  const renderMiddle = () => {
    if (tab === "models" && activeKey === "defaults") {
      return (
        <FieldGroup title="defaults">
          <ConfigFields value={item} onChange={setItem} />
        </FieldGroup>
      );
    }

    if (tab === "models") {
      return (
        <>
          {!item.path && <p className="text-xs text-amber-400">未填写 path，启动时将跳过该模型</p>}
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            <div className="flex flex-col gap-1 min-w-0">
              <span className="font-mono text-xs text-slate-400">modelName</span>
              <span className="truncate rounded border border-slate-800 bg-slate-950 px-2 py-1.5 font-mono text-xs text-blue-300">
                {activeKey}
              </span>
            </div>
            <ValueField name="path" value={item.path} options={[]} plain onChange={(v) => setItem({ ...item, path: v })} />
            <ValueField
              name="engine"
              value={item.engine}
              options={FIELD_OPTIONS.engine}
              plain={false}
              onChange={(v) => setItem({ ...item, engine: v })}
            />
            <ValueField
              name="launch"
              value={item.launch}
              options={FIELD_OPTIONS.launch}
              plain={false}
              onChange={(v) => setItem({ ...item, launch: v })}
            />
          </div>
          <FieldGroup title="load">
            <ConfigFields value={item.load ?? {}} onChange={(load) => setItem({ ...item, load })} />
          </FieldGroup>
        </>
      );
    }

    return (
      <>
        {isDefault && (
          <FieldGroup title="team">
            <ConfigFields
              value={item.team ?? {}}
              extraOptions={{ mian: modelIds, sub: modelIds }}
              onChange={(team) => setItem({ ...item, team })}
            />
          </FieldGroup>
        )}
        <FieldGroup title="load">
          <ConfigFields value={item.load ?? {}} onChange={(load) => setItem({ ...item, load })} />
        </FieldGroup>
      </>
    );
  };

  return (
    <div className="space-y-4">
      <div className="flex gap-2 border-b border-slate-800 pb-3">
        {(Object.keys(TAB_LABELS) as Tab[]).map((id) => (
          <button
            key={id}
            onClick={() => { setTab(id); setSelected(""); setTip(""); }}
            className={`px-4 py-1.5 rounded-lg text-sm font-medium ${
              tab === id ? "bg-blue-600 text-white" : "bg-slate-800 text-slate-400"
            }`}
          >
            {TAB_LABELS[id]}
          </button>
        ))}
      </div>

      <div className="flex items-start gap-4">
        {/* 左：菜单 */}
        <aside className="w-52 shrink-0 space-y-3 rounded-lg border border-slate-800 bg-slate-900/60 p-3">
          <div className="px-1 text-xs font-semibold text-slate-400">{tab === "models" ? "models.json" : "modelSetting.json"}</div>
          <nav className="max-h-[60vh] space-y-1 overflow-y-auto">
            {menu.map((m) => {
              const active = m.key === activeKey;
              const color = m.alert ? "text-rose-400" : active ? "text-white" : "text-slate-300";
              const label = m.key === DEFAULT_KEY[tab] ? "默认设置" : m.key;
              return (
                <button
                  key={m.key}
                  onClick={() => setSelected(m.key)}
                  title={m.key}
                  className={`block w-full truncate rounded-lg px-3 py-1.5 text-left font-mono text-sm ${color} ${
                    active ? "bg-slate-800" : "hover:bg-slate-800"
                  }`}
                >
                  {label}
                </button>
              );
            })}
          </nav>
          <div className="flex flex-col gap-2 border-t border-slate-800 pt-3">
            <input
              list={tab === "settings" ? "engine-options" : undefined}
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && addItem()}
              placeholder={tab === "models" ? "新模型 ID" : "引擎名（可下拉选择）"}
              className="min-w-0 rounded-lg border border-slate-800 bg-slate-900 px-3 py-1.5 text-sm focus:border-blue-500 focus:outline-none"
            />
            <datalist id="engine-options">
              {ENGINE_OPTIONS.map((o) => <option key={o} value={o} />)}
            </datalist>
            <button
              onClick={addItem}
              className="flex items-center justify-center gap-1.5 rounded-lg bg-emerald-600 px-3 py-1.5 text-sm hover:bg-emerald-500"
            >
              <Plus size={15} />
              <span>{tab === "models" ? "新增模型（拷贝 template）" : "新增引擎（拷贝 template）"}</span>
            </button>
          </div>
        </aside>

        {/* 中：所选项的数据 */}
        <section className="min-w-0 flex-1 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="font-mono text-sm font-bold text-blue-300">
              {isDefault ? "默认设置" : activeKey}
            </h3>
            {!isDefault && (
              <button onClick={removeItem} className="text-slate-500 hover:text-rose-400" title="删除">
                <Trash2 size={15} />
              </button>
            )}
          </div>
          {renderMiddle()}
        </section>

        {/* 右：generation + 保存 */}
        <aside className="sticky top-0 w-96 shrink-0 space-y-3 rounded-lg border border-slate-800 bg-slate-900/60 p-3">
          <div className="flex items-center justify-between gap-2">
            <span className="font-mono text-xs font-bold text-blue-300">generation</span>
            {tip && <span className={`text-xs ${tip.startsWith("❌") ? "text-rose-400" : "text-emerald-400"}`}>{tip}</span>}
          </div>
          {tab === "models" && activeKey === "defaults" ? (
            <p className="text-xs text-slate-500">defaults 不包含 generation 设定</p>
          ) : (
            <ConfigFields value={item.generation ?? {}} onChange={(generation) => setItem({ ...item, generation })} />
          )}
          <button
            onClick={handleSave}
            className="flex w-full items-center justify-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium hover:bg-blue-500"
          >
            <Save size={15} />
            <span>保存 {FILE_NAMES[tab]}</span>
            {dirty && <span className="text-xs text-amber-300">· 未保存</span>}
          </button>
        </aside>
      </div>
    </div>
  );
};
