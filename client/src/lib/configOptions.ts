/** 模型配置界面的候选值与字段说明：候选值只作下拉提示，实际值仍可手动输入 */
const BOOL_OPTIONS = ["true", "false"];

export const FIELD_OPTIONS: Record<string, string[]> = {
  engine: ["llama", "vllm", "sglang", "ninfer"],
  launch: BOOL_OPTIONS,
  // 后端目前仅保存该字段，取值待确认
  data_compression: ["default", "none"],
  dtype: ["bfloat16", "float16", "float32", "auto", "fp8", "int8", "bf16", "f16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0"],
  kv_dtype: ["bf16", "int8"],
  useVram: ["2048", "4096", "6144", "8192", "12288", "16384", "24576", "32768"],
  batch_size: ["128", "256", "512", "1024", "2048"],
  parallel_slots: ["1", "2", "4", "8"],
  gpu_offload_layers: ["-1", "0", "16", "32", "48", "99"],
  tensor_parallel: ["1", "2", "4", "8"],
  draft_tokens: ["1", "2", "3", "4", "6"],
  mtp: BOOL_OPTIONS,
  flash_attention: BOOL_OPTIONS,
  use_mlock: BOOL_OPTIONS,
  enable_sleep_mode: BOOL_OPTIONS,
  enable_memory_saver: BOOL_OPTIONS,
  trust_remote_code: BOOL_OPTIONS,
  lm_head_draft: BOOL_OPTIONS,
  gpu_memory_utilization: ["0.7", "0.8", "0.85", "0.9", "0.95"],
  temperature: ["0", "0.3", "0.5", "0.7", "1"],
  top_p: ["0.8", "0.9", "0.95", "1"],
  top_k: ["0", "20", "40", "50", "100"],
  min_p: ["0", "0.05", "0.1"],
  repetition_penalty: ["1", "1.05", "1.1", "1.2"],
  presence_penalty: ["0", "0.5", "1"],
  frequency_penalty: ["0", "0.5", "1"],
  mirostat: ["0", "1", "2"],
  max_tokens: ["-1", "1024", "2048", "4096", "8192", "16384"],
  sleepTime: ["0", "60", "300", "600", "1800"],
  host: ["0.0.0.0", "127.0.0.1"],
  api_port: ["8666"],
};

export const ENGINE_OPTIONS = FIELD_OPTIONS.engine;

/** 始终按文本保存的字段，避免 "123456" 这类字符串被误转成数字 */
export const TEXT_FIELDS = new Set([
  "path", "engine", "data_compression", "dtype", "kv_dtype", "lora", "host", "api_key", "system_prompt",
]);

/** 字段与分组的中文说明，显示在字段名旁 */
export const FIELD_HINTS: Record<string, string> = {
  // models.json
  defaults: "全局默认",
  server: "服务配置",
  host: "监听地址",
  api_port: "API 端口",
  api_key: "API 密钥（留空不鉴权）",
  sleepTime: "空闲休眠秒数（0 为不自动休眠）",
  path: "模型文件或目录",
  engine: "推理引擎（留空自动识别）",
  launch: "启动时自动加载",
  data_compression: "数据压缩策略",
  estimated_vram_mb: "估算显存（自动写入）",
  // 加载与生成参数
  load: "加载参数",
  generation: "生成参数",
  mtp: "MTP 投机解码",
  lora: "LoRA 路径",
  draft: "草稿模型路径列表",
  useVram: "显存/上下文预算（MB）",
  batch_size: "批大小",
  parallel_slots: "并行槽位",
  gpu_offload_layers: "GPU 卸载层数（-1 为全部）",
  tensor_parallel: "张量并行卡数",
  draft_tokens: "草稿 token 数",
  dtype: "数据类型",
  temperature: "温度",
  top_p: "Top-P",
  top_k: "Top-K",
  repetition_penalty: "重复惩罚",
  max_tokens: "最大生成 token（-1 为按剩余上下文动态）",
  stop: "停止词",
  system_prompt: "系统提示词",
  // modelSetting.json
  template: "默认设定",
  team: "组合加载",
  mian: "主模型组",
  sub: "子模型组",
  flash_attention: "Flash Attention",
  use_mlock: "锁定内存（mlock）",
  min_p: "Min-P",
  mirostat: "Mirostat 模式",
  gpu_memory_utilization: "显存占用上限",
  enable_sleep_mode: "休眠时权重移至内存",
  enable_memory_saver: "休眠时权重移至内存",
  trust_remote_code: "信任远程代码",
  presence_penalty: "存在惩罚",
  frequency_penalty: "频率惩罚",
  kv_dtype: "KV 缓存精度",
  lm_head_draft: "LM Head 草稿",
};
