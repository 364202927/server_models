import React, { useState } from "react";
import { Send } from "lucide-react";
import { useAppStore } from "../store/useAppStore";
import { gatewayBase, gatewayHeaders } from "../lib/gateway";

export const ModelTestPage: React.FC = () => {
  const { models, settings, remoteModelIds } = useAppStore();
  // 优先用网关实际可用的模型，本地配置仅作回退
  const modelIds = remoteModelIds.length > 0 ? remoteModelIds : Object.keys(models || {});
  const [pickedModel, setSelectedModel] = useState("");
  const selectedModel = modelIds.includes(pickedModel) ? pickedModel : modelIds[0] || "";
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Array<{ role: string; content: string }>>([]);
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    const targetModel = selectedModel || modelIds[0];
    if (!input.trim() || !targetModel || loading) return;

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
        }),
      });
      const data = await res.json();
      const reply = data?.choices?.[0]?.message?.content || JSON.stringify(data);
      setMessages([...nextMessages, { role: "assistant", content: reply }]);
    } catch (err) {
      setMessages([...nextMessages, { role: "assistant", content: `[请求失败]: ${err}` }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full space-y-3">
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
          onClick={() => setMessages([])}
          className="text-xs px-2.5 py-1 bg-slate-800 hover:bg-slate-700 rounded text-slate-300"
        >
          清空对话
        </button>
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
  );
};