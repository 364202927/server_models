/** 模型配置的三层继承：template < 引擎默认 < 本模型（与后端 model_spec._merged 保持一致） */
type Json = Record<string, any>;
export type Section = "load" | "generation";

/** 模型所用引擎，未填写时与后端一致回退 llama */
export const engineOf = (model: Json | undefined): string => model?.engine || "llama";

/** 本模型之外的继承值：template.<section> + <engine>.<section> */
export const inheritedOf = (settings: Json, engine: string, section: Section): Json => ({
  ...(settings.template?.[section] ?? {}),
  ...(settings[engine]?.[section] ?? {}),
});

/** 界面展示值：继承值叠加本模型已覆盖的值 */
export const effectiveOf = (settings: Json, model: Json | undefined, section: Section): Json => ({
  ...inheritedOf(settings, engineOf(model), section),
  ...(model?.[section] ?? {}),
});

/** 保存时只写与继承值不同的键，或本模型原本就已覆盖的键，避免把默认值整份拷进 models.json */
export const overridesOf = (next: Json, inherited: Json, current: Json = {}): Json =>
  Object.fromEntries(
    Object.entries(next).filter(
      ([key, value]) => key in current || JSON.stringify(value) !== JSON.stringify(inherited[key]),
    ),
  );

const isEmpty = (v: unknown): boolean => v === "" || v === null || v === undefined || (Array.isArray(v) && v.length === 0);

/** 去掉空值，随请求下发的生成参数 */
export const nonEmpty = (params: Json): Json =>
  Object.fromEntries(Object.entries(params).filter(([, v]) => !isEmpty(v)));
