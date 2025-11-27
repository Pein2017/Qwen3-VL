# Design: Dynamic Hard-Sample Mining (token_acc, post-70% activation)

## Goals
- 使用 **token_acc**（越低越难）作为唯一难度信号；EMA 聚合，<3 次观测用均值。
- 保持目标集总长度不变，后 30% 训练阶段将目标样本重排为 30% hard（可重复）+ 70% 全量有放回的常规样本。
- 源数据集按可配置比例 (`source_ratio`，默认 8% 目标长度) 追加到调度，不参与挖掘/重权。
- 仍兼容 ms-swift/transformers Trainer、DeepSpeed ZeRO-2；最小化对 Trainer 的侵入；rank0 选池并广播 schedule。

## Flow
1) **采集**：在 `compute_loss` 包装中计算 per-sample token_acc（logits argmax vs labels，mask -100），rank0 聚合并写入 `HardSampleTracker`（sample_id, dataset, base_idx）。
2) **激活条件**：训练进度 < `activate_after_pct`（默认 0.7）仅记录，不重排；达阈值后每轮执行挖掘。
3) **选池**：epoch 末在目标集上按 EMA( token_acc ) 由低到高排序，取前 `hard_pool_frac`（默认 0.3）作为 hard_pool。
4) **重建调度**：
   - 目标：生成长度等于原目标池的序列，组成 30% hard（有放回）+ 70% 目标全量有放回常规样本。
   - 源：追加数量为 `round(source_ratio * len(target))` 的源样本（从所有源池均匀有放回），与目标条目合并并打乱；源比例固定，不受挖掘影响。
   - 将结果通过 `set_external_hsm_schedule` 注入数据集，DataLoader 以确定性 sampler 迭代该顺序。
5) **日志**：每轮记录 `hsm/` 前缀的池指标（hard/regular acc mean/p90、hard_seen/regular_seen、hard_pool_coverage、hard_pool_size、schedule_len、mining_active 等）。

## Integration Points
- **Config**：`custom.hard_sample_mining` 使用新字段：`enabled`, `start_epoch`, `hard_pool_frac`, `activate_after_pct`, `source_ratio`, `ema_decay`, `log_pool_metrics`, `log_prefix`。旧模式字段已移除。
- **Trainer**：在初始化时 attach 采集包装；训练 dataloader 改为确定性 sampler（无 shuffle），以尊重外部 schedule 顺序；rank0 选池并通过 broadcast 同步 schedule。
- **Dataset**：
  - `BaseCaptionDataset` / `FusionCaptionDataset` 支持 `set_external_hsm_schedule`，当存在时 `__len__`/`__getitem__` 直接使用外部顺序；不再保留 legacy 计划接口。
  - Fusion：源/目标池保持原始内容；源 quota 在回调中按 `source_ratio` 计算。

## Non-Goals / Removed
- 不再支持基于 loss 的固定 top-K / target_epoch_size 缩容；不做 plateau 触发；不做 mine_clean/recompute_full_pass。

## Edge Cases
- 小数据集：若 hard_pool 为空或不足，允许有放回补齐；仍保持目标长度。
- DDP/ZeRO-2：仅 rank0 聚合/排序；计划通过 all_gather_object / broadcast 同步；Sampler 使用 epoch-seeded RNG 确保一致顺序。
