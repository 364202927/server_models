import React, { useEffect, useId, useState } from "react";
import { Plus, Trash2, X } from "lucide-react";
import { FIELD_HINTS, FIELD_OPTIONS, TEXT_FIELDS } from "../lib/configOptions";

export type Json = Record<string, any>;
type FieldKind = "list" | "text" | "scalar";

export const isPlainObject = (v: unknown): v is Json =>
  typeof v === "object" && v !== null && !Array.isArray(v);

/** 字段类型由初始值决定；空值的标量交给输入文本自动识别 */
const kindOf = (key: string, initial: unknown): FieldKind => {
  if (Array.isArray(initial)) return "list";
  if (TEXT_FIELDS.has(key) || (typeof initial === "string" && initial !== "")) return "text";
  return "scalar";
};

const toText = (value: unknown): string => {
  if (Array.isArray(value)) return value.join(", ");
  return value === undefined || value === null ? "" : String(value);
};

/** 文本还原为 JSON 值：列表按逗号切分，标量识别 true/false 与数字 */
const fromText = (text: string, kind: FieldKind): unknown => {
  if (kind === "list") return text.split(",").map((s) => s.trim()).filter(Boolean);
  if (kind === "text") return text;
  const t = text.trim();
  if (t === "true" || t === "false") return t === "true";
  return t !== "" && Number.isFinite(Number(t)) ? Number(t) : text;
};

interface ValueFieldProps {
  name: string;
  value: unknown;
  options: string[];
  plain: boolean;
  onChange: (next: unknown) => void;
  onRemove?: () => void;
}

/** 单个字段：除 plain 外均为下拉可输入（input + datalist） */
export const ValueField: React.FC<ValueFieldProps> = ({ name, value, options, plain, onChange, onRemove }) => {
  const inputId = useId();
  const listId = `${inputId}-list`;
  const [kind] = useState(() => kindOf(name, value));
  const [text, setText] = useState(() => toText(value));
  const hint = FIELD_HINTS[name];
  const hasList = !plain && options.length > 0;

  // 外部值变化时同步文本；若文本已解析为当前值（如输入中的 "1."），保留用户输入
  useEffect(() => {
    if (JSON.stringify(fromText(text, kind)) !== JSON.stringify(value)) setText(toText(value));
  }, [value]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleInput = (raw: string) => {
    setText(raw);
    onChange(fromText(raw, kind));
  };

  return (
    <div className="flex flex-col gap-1 min-w-0">
      <div className="flex items-center justify-between gap-2">
        <label htmlFor={inputId} className="truncate font-mono text-xs text-slate-400" title={hint}>
          {name}
          {hint && <span className="ml-2 font-sans text-slate-600">{hint}</span>}
        </label>
        {onRemove && (
          <button onClick={onRemove} className="text-slate-600 hover:text-rose-400" title="删除字段">
            <X size={12} />
          </button>
        )}
      </div>
      <input
        id={inputId}
        list={hasList ? listId : undefined}
        value={text}
        onChange={(e) => handleInput(e.target.value)}
        placeholder={plain ? "" : kind === "list" ? "多个值用逗号分隔" : "可下拉选择，也可直接输入"}
        className="w-full min-w-0 rounded border border-slate-800 bg-slate-950 px-2 py-1.5 font-mono text-xs text-slate-200 focus:border-blue-500 focus:outline-none"
      />
      {hasList && (
        <datalist id={listId}>
          {options.map((o) => <option key={o} value={o} />)}
        </datalist>
      )}
    </div>
  );
};

interface FieldGroupProps {
  title: string;
  onRemove?: () => void;
  children: React.ReactNode;
}

/** 嵌套对象的分组卡片 */
export const FieldGroup: React.FC<FieldGroupProps> = ({ title, onRemove, children }) => {
  const hint = FIELD_HINTS[title];
  return (
    <div className="min-w-0 space-y-3 rounded-lg border border-slate-800 bg-slate-900/60 p-3">
      <div className="flex items-center justify-between gap-2">
        <span className="font-mono text-xs font-bold text-blue-300">
          {title}
          {hint && <span className="ml-2 font-sans font-normal text-slate-500">{hint}</span>}
        </span>
        {onRemove && (
          <button onClick={onRemove} className="text-slate-500 hover:text-rose-400" title="删除">
            <Trash2 size={14} />
          </button>
        )}
      </div>
      {children}
    </div>
  );
};

interface ConfigFieldsProps {
  value: Json;
  onChange: (next: Json) => void;
  /** 仅允许自由输入的字段，其余字段均为下拉可输入 */
  plainKeys?: string[];
  /** 额外候选值，优先于 FIELD_OPTIONS */
  extraOptions?: Record<string, string[]>;
}

/** 按 JSON 结构递归渲染：嵌套对象成分组，其余成字段；以下划线开头的键（如 _comment）不展示但原样保留。列数按容器宽度变化 */
export const ConfigFields: React.FC<ConfigFieldsProps> = ({ value, onChange, plainKeys = [], extraOptions = {} }) => {
  const [newKey, setNewKey] = useState("");

  const setKey = (key: string, next: unknown) => onChange({ ...value, [key]: next });

  const removeKey = (key: string) => {
    const next = { ...value };
    delete next[key];
    onChange(next);
  };

  const addKey = () => {
    const key = newKey.trim();
    if (!key || key in value) return;
    onChange({ ...value, [key]: "" });
    setNewKey("");
  };

  return (
    <div className="@container min-w-0">
      <div className="grid grid-cols-1 gap-3 @md:grid-cols-2 @3xl:grid-cols-3">
        {Object.entries(value)
          .filter(([key]) => !key.startsWith("_"))
          .map(([key, child]) =>
            isPlainObject(child) ? (
              <div key={key} className="col-span-full">
                <FieldGroup title={key} onRemove={() => removeKey(key)}>
                  <ConfigFields
                    value={child}
                    onChange={(next) => setKey(key, next)}
                    plainKeys={plainKeys}
                    extraOptions={extraOptions}
                  />
                </FieldGroup>
              </div>
            ) : (
              <ValueField
                key={key}
                name={key}
                value={child}
                options={extraOptions[key] ?? FIELD_OPTIONS[key] ?? []}
                plain={plainKeys.includes(key)}
                onChange={(next) => setKey(key, next)}
                onRemove={plainKeys.includes(key) ? undefined : () => removeKey(key)}
              />
            ),
          )}

        <div className="col-span-full flex gap-2">
          <input
            value={newKey}
            onChange={(e) => setNewKey(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && addKey()}
            placeholder="新增字段名"
            className="min-w-0 flex-1 rounded border border-slate-800 bg-slate-950 px-2 py-1.5 font-mono text-xs text-slate-200 focus:border-blue-500 focus:outline-none"
          />
          <button
            onClick={addKey}
            className="flex shrink-0 items-center gap-1 rounded bg-slate-800 px-2.5 py-1.5 text-xs text-slate-300 hover:bg-slate-700"
          >
            <Plus size={13} />
            <span>添加字段</span>
          </button>
        </div>
      </div>
    </div>
  );
};
