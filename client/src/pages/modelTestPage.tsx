import React, { useEffect, useRef, useState } from "react";
import { Save, Send } from "lucide-react";
import { useAppStore } from "../store/useAppStore";
import { gatewayBase, gatewayHeaders } from "../lib/gateway";
import { ConfigFields, type Json } from "../components/configFields";
import { effectiveOf, engineOf, inheritedOf, nonEmpty, overridesOf } from "../lib/modelConfig";

/** 含派生子代理工具名 task，classifier 据此判为主会话 */
const MAIN_ROUTE_TOOLS = [
  {
    type: "function",
    function: { name: "task", description: "spawn subagent", parameters: { type: "object", properties: {} } },
  },
];

export const ModelTestPage: React.FC = () => {
  const { models, settings, remoteModelIds, saveLocalConfig } = useAppStore();
  // 优先用网关实际可用的模型，本地配置仅作回退
  const modelIds = remoteModelIds.length > 0 ? remoteModelIds : Object.keys(models?.models || {});
  const [pickedModel, setSelectedModel] = useState("");
  const selectedModel = modelIds.includes(pickedModel) ? pickedModel : modelIds[0] || "";
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Array<{ role: string; content: string }>>([]);
  const [loading, setLoading] = useState(false);
  const [genParams, setGenParams] = useState<Json>({});
  const [tip, setTip] = useState("");
  // 勾选后请求带上派生子代理工具，后端 classifier 才会判为主路（否则默认走支路）
  const [mainRoute, setMainRoute] = useState(true);
  // 对话代次：清空 / 切页 / 切模型时自增，作废仍在途的请求，避免旧回复写回新对话
  const epoch = useRef(0);

  const resetChat = () => {
    epoch.current += 1;
    setMessages([]);
    setLoading(false);
  };
  // 离开页面（组件卸载）时同样作废在途请求
  useEffect(() => () => { epoch.current += 1; }, []);
  // 换模型等于换会话，旧对话不再属于它
  useEffect(resetChat, [selectedModel]);

  // 当前模型的 generation：template < 引擎默认 < 本模型；切换模型或配置刷新时重置
  const modelCfg: Json | undefined = models?.models?.[selectedModel];
  useEffect(() => {
    setGenParams(effectiveOf(settings || {}, modelCfg, "generation"));
    setTip("");
  }, [selectedModel, settings, models]); // eslint-disable-line react-hooks/exhaustive-deps

  const saveGenParams = async () => {
    if (!modelCfg) {
      setTip("❌ 本地 models.json 中没有该模型");
      return;
    }
    const inherited = inheritedOf(settings || {}, engineOf(modelCfg), "generation");
    const next = {
      ...models,
      models: {
        ...models.models,
        [selectedModel]: { ...modelCfg, generation: overridesOf(genParams, inherited, modelCfg.generation) },
      },
    };
    setTip((await saveLocalConfig("models.json", next)) ? "✅ 已保存" : "❌ 保存失败");
  };

  const sendMessage = async () => {
    const targetModel = selectedModel || modelIds[0];
    if (!input.trim() || !targetModel || loading) return;

    const myEpoch = epoch.current;
    const nextMessages = [...messages, { role: "user", content: input.trim() }];
    setMessages(nextMessages);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch(`${gatewayBase(settings)}/v1/chat/completions`, {
        method: "POST",
        headers: gatewayHeaders(settings),
        body: JSON.stringify({
          message_id: 0,
          model: targetModel,
          messages: nextMessages,
          stream: false,
          ...(mainRoute ? { tools: MAIN_ROUTE_TOOLS } : {}),
          // 生成参数放在请求顶层，后端会作为本次推理的覆盖参数
          ...nonEmpty(genParams),
        }),
      });
      const data = await res.json();
      if (myEpoch !== epoch.current) return;
      const reply = data?.choices?.[0]?.message?.content || JSON.stringify(data);
      setMessages([...nextMessages, { role: "assistant", content: reply }]);
    } catch (err) {
      if (myEpoch !== epoch.current) return;
      setMessages([...nextMessages, { role: "assistant", content: `[请求失败]: ${err}` }]);
    } finally {
      if (myEpoch === epoch.current) setLoading(false);
    }
  };

  return (
    <div className="flex h-full gap-4">
      <aside className="flex w-80 shrink-0 flex-col gap-3 overflow-y-auto rounded-lg border border-slate-800 bg-slate-900/60 p-3">
        <div className="flex items-center justify-between gap-2">
          <span className="font-mono text-xs font-bold text-blue-300">generation</span>
          {tip && <span className={`text-xs ${tip.startsWith("❌") ? "text-rose-400" : "text-emerald-400"}`}>{tip}</span>}
        </div>
        {selectedModel ? (
          <ConfigFields key={selectedModel} removable={false} value={genParams} onChange={setGenParams} />
        ) : (
          <p className="text-xs text-slate-500">暂无可用模型</p>
        )}
        <button
          onClick={saveGenParams}
          disabled={!selectedModel}
          className="flex items-center justify-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium hover:bg-blue-500 disabled:bg-slate-700"
        >
          <Save size={15} />
          <span>保存 gen 参数</span>
        </button>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col space-y-3">
      <div className="flex items-center gap-3">
        <span className="text-xs text-slate-400">选择测试模型:</span>
        <select
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
          className="bg-slate-900 border border-slate-800 rounded-lg px-3 py-1.5 text-sm text-slate-200"
        >
          {modelIds.map((id) => (
            <option key={id} value={id}>{id}</option>
          ))}
        </select>
        <button
          onClick={resetChat}
          className="text-xs px-2.5 py-1 bg-slate-800 hover:bg-slate-700 rounded text-slate-300"
        >
          清空对话
        </button>
        <label className="flex items-center gap-1.5 text-xs text-slate-400" title="请求带上 task 工具，后端判为主路；取消勾选则走支路">
          <input type="checkbox" checked={mainRoute} onChange={(e) => { setMainRoute(e.target.checked); resetChat(); }} />
          主路
        </label>
      </div>

      <div className="flex-1 bg-slate-900 border border-slate-800 rounded-lg p-4 overflow-y-auto space-y-3 select-text">
        {messages.map((m, i) => (
          <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
            <div
              className={`max-w-2xl px-3.5 py-2 rounded-lg text-sm whitespace-pre-wrap ${
                m.role === "user" ? "bg-blue-600 text-white" : "bg-slate-800 text-slate-200"
              }`}
            >
              {m.content}
            </div>
          </div>
        ))}
      </div>

      <div className="flex gap-2">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          placeholder="输入测试内容，按 Enter 发送..."
          className="flex-1 bg-slate-900 border border-slate-800 rounded-lg px-3.5 py-2 text-sm focus:outline-none focus:border-blue-500"
        />
        <button
          onClick={sendMessage}
          disabled={loading}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 rounded-lg text-sm font-medium flex items-center gap-1.5"
        >
          <Send size={15} />
          <span>{loading ? "推理中..." : "发送"}</span>
        </button>
      </div>
      </div>
    </div>
  );
};
