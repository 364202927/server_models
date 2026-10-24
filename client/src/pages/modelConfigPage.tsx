import React, { useEffect, useState } from "react";
import { FolderOpen, Play, Power, Save, Plus, Trash2 } from "lucide-react";
import { useAppStore } from "../store/useAppStore";
import { ConfigFields, FieldGroup, ValueField, isPlainObject, type Json } from "../components/configFields";
import { ENGINE_OPTIONS, FIELD_OPTIONS } from "../lib/configOptions";
import { effectiveOf, engineOf, inheritedOf, overridesOf, type Section } from "../lib/modelConfig";
import { isTauri, pickDirectory } from "../lib/bridge";
import { ADMIN, gatewayState } from "../lib/gateway";

type Tab = "models" | "settings";

const TAB_LABELS: Record<Tab, string> = { models: "配置", settings: "设定" };
const SECTIONS: Section[] = ["load", "generation"];
const FILE_NAMES: Record<Tab, string> = { models: "models.json", settings: "modelSetting.json" };

/** 菜单项：key 对应数据键，running 为绿字高亮（后端状态 RUNNING） */
interface MenuItem {
  key: string;
  running?: boolean;
}

/** 首项“默认设置”在 models.json 对应 defaults，在 modelSetting.json 对应 template */
const DEFAULT_KEY: Record<Tab, string> = { models: "defaults", settings: "template" };

/** 已占用显存的状态：RUNNING 或休眠驻留，均允许卸载 */
const LOADED_STATES = new Set(["RUNNING", "SLEEPING_RAM"]);

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
  const { settings, models, processes, saveLocalConfig, sendAdminCmd, pollStatusAndLogs } = useAppStore();
  const [tab, setTab] = useState<Tab>("models");
  const [selected, setSelected] = useState("");
  const [draftModels, setDraftModels] = useState<Json>({});
  const [draftSettings, setDraftSettings] = useState<Json>({});
  const [newName, setNewName] = useState("");
  const [tip, setTip] = useState("");
  const [runBusy, setRunBusy] = useState(false);

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

  /** 后端 1001 回包里的模型状态，未加载或网关不可达时为 UNLOADED */
  const stateOf = (id: string): string => processes.find((p) => p.model === id)?.state ?? "UNLOADED";
  const isLoaded = (id: string) => LOADED_STATES.has(stateOf(id));

  // 左侧菜单：配置页为 defaults + models.json 中的模型（运行中绿字），设定页为 template + 引擎
  const menu: MenuItem[] = tab === "models"
    ? [{ key: "defaults" }, ...modelIds.map((id) => ({ key: id, running: stateOf(id) === "RUNNING" }))]
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

  /** 删除模型立即写回 models.json（连同草稿里其他未保存的修改一并落盘） */
  const removeItem = async () => {
    if (tab === "models") {
      const next = { ...modelMap };
      delete next[activeKey];
      const nextModels = { ...draftModels, models: next };
      setDraftModels(nextModels);
      setSelected("");
      const ok = await saveLocalConfig("models.json", nextModels);
      setTip(ok ? "✅ 已从 models.json 删除" : "❌ 删除后写入 models.json 失败");
      return;
    }
    const next = { ...draftSettings };
    delete next[activeKey];
    setDraftSettings(next);
    setSelected("");
  };

  /** 激活 / 卸载当前模型，指令返回后立即刷新一次状态 */
  const toggleModelRun = async () => {
    const unload = isLoaded(activeKey);
    setRunBusy(true);
    try {
      const res = await sendAdminCmd(unload ? ADMIN.unload : ADMIN.ensureLoaded, null, activeKey);
      setTip(res ? (unload ? "✅ 已卸载模型" : "✅ 模型已激活") : `❌ 模型操作失败: ${gatewayState.lastError}`);
      await pollStatusAndLogs();
    } finally {
      setRunBusy(false);
    }
  };

  const choosePath = async () => {
    if (!isTauri()) {
      setTip("❌ 目录选择仅在桌面端可用");
      return;
    }
    try {
      const dir = await pickDirectory();
      if (dir) setItem({ ...item, path: dir });
    } catch (err) {
      setTip(`❌ 选择目录失败: ${err}`);
    }
  };

  const isModel = tab === "models" && activeKey !== "defaults";

  /** load / generation 分组：模型项展示“template + 引擎”继承后的全部可选项，其余项展示自身数据 */
  const renderSection = (section: Section) => {
    if (!isModel) {
      return (
        <FieldGroup title={section}>
          <ConfigFields value={item[section] ?? {}} onChange={(next) => setItem({ ...item, [section]: next })} />
        </FieldGroup>
      );
    }
    const inherited = inheritedOf(draftSettings, engineOf(item), section);
    return (
      <FieldGroup title={section}>
        <ConfigFields
          removable={false}
          value={effectiveOf(draftSettings, item, section)}
          onChange={(next) => setItem({ ...item, [section]: overridesOf(next, inherited, item[section]) })}
        />
      </FieldGroup>
    );
  };

  const renderBasics = () => {
    if (!isModel) return null;
    return (
      <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
        <ValueField name="engine" value={item.engine} options={FIELD_OPTIONS.engine} plain={false} onChange={(v) => setItem({ ...item, engine: v })} />
        <ValueField name="launch" value={item.launch} options={FIELD_OPTIONS.launch} plain={false} onChange={(v) => setItem({ ...item, launch: v })} />
        <ValueField name="data_compression" value={item.data_compression} options={FIELD_OPTIONS.data_compression} plain={false} onChange={(v) => setItem({ ...item, data_compression: v })} />
        <div className="md:col-span-3">
          <ValueField
            name="path"
            value={item.path}
            options={[]}
            plain
            onChange={(v) => setItem({ ...item, path: v })}
            trailing={
              <button
                onClick={choosePath}
                title="选择模型目录"
                className="flex shrink-0 items-center gap-1 rounded border border-slate-700 bg-slate-800 px-2 py-1.5 text-xs text-slate-300 hover:bg-slate-700"
              >
                <FolderOpen size={14} />
                <span>选择目录</span>
              </button>
            }
          />
        </div>
      </div>
    );
  };

  /** 标题旁的模型开关：RUNNING 显示卸载，其余显示激活，加载/请求中禁用 */
  const renderRunButton = () => {
    if (!isModel) return null;
    const loaded = isLoaded(activeKey);
    const loading = stateOf(activeKey) === "LOADING";
    return (
      <button
        onClick={toggleModelRun}
        disabled={runBusy || loading}
        className={`flex shrink-0 items-center gap-1 rounded-lg px-2.5 py-1 text-xs font-medium text-white disabled:opacity-50 ${
          loaded ? "bg-rose-600 hover:bg-rose-500" : "bg-emerald-600 hover:bg-emerald-500"
        }`}
      >
        {loaded ? <Power size={13} /> : <Play size={13} />}
        <span>{runBusy ? "处理中…" : loading ? "加载中…" : loaded ? "卸载模型" : "激活模型"}</span>
      </button>
    );
  };

  const renderBody = () => {
    if (tab === "models" && activeKey === "defaults") {
      return (
        <FieldGroup title="defaults">
          <ConfigFields value={item} onChange={setItem} />
        </FieldGroup>
      );
    }
    return (
      <>
        {isModel && !item.path && <p className="text-xs text-amber-400">未填写 path，启动时将跳过该模型</p>}
        {renderBasics()}
        {tab === "settings" && isDefault && (
          <FieldGroup title="team">
            <ConfigFields
              value={item.team ?? {}}
              extraOptions={{ mian: modelIds, sub: modelIds }}
              onChange={(team) => setItem({ ...item, team })}
            />
          </FieldGroup>
        )}
        {/* load 与 generation 并排等宽，字段网格随容器宽度自适应 */}
        <div className="grid grid-cols-1 items-start gap-4 xl:grid-cols-2">
          {SECTIONS.map((section) => <React.Fragment key={section}>{renderSection(section)}</React.Fragment>)}
        </div>
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
          <div className="px-1 text-xs font-semibold text-slate-400">{FILE_NAMES[tab]}</div>
          <nav className="max-h-[60vh] space-y-1 overflow-y-auto">
            {menu.map((m) => {
              const active = m.key === activeKey;
              const color = m.running ? "text-emerald-400" : active ? "text-white" : "text-slate-300";
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

        {/* 右：标题栏（保存/删除）+ 所选项数据 */}
        <section key={`${tab}:${activeKey}`} className="min-w-0 flex-1 space-y-4">
          <div className="flex items-center justify-between gap-3">
            <div className="flex min-w-0 items-center gap-3">
              <h3 className="truncate font-mono text-sm font-bold text-blue-300">
                {isDefault ? "默认设置" : activeKey}
              </h3>
              {renderRunButton()}
            </div>
            <div className="flex shrink-0 items-center gap-3">
              {tip && <span className={`text-xs ${tip.startsWith("❌") ? "text-rose-400" : "text-emerald-400"}`}>{tip}</span>}
              {!isDefault && (
                <button onClick={removeItem} className="text-slate-500 hover:text-rose-400" title="删除">
                  <Trash2 size={15} />
                </button>
              )}
              <button
                onClick={handleSave}
                className="flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-1.5 text-sm font-medium hover:bg-blue-500"
              >
                <Save size={15} />
                <span>保存 {FILE_NAMES[tab]}</span>
                {dirty && <span className="text-xs text-amber-300">· 未保存</span>}
              </button>
            </div>
          </div>
          {renderBody()}
        </section>
      </div>
    </div>
  );
};
