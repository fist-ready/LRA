# 当前交接：借阅事件时序模型待用户启动全量训练

已完成事件构造、图书与读者编码、联合训练、模型加载及正式 Top-10 接口。三项定向测试与真实小样本训练通过。用户启动 run_event_training.ps1 --suite 后，请从 outputs/event_sequence_v1/runs 读取权重与 manifest，再开始开发/诊断评测；暂不以训练损失代替官方指标。完整命令见 EVENT_SEQUENCE_TRAINING.md。

# 当前提交版：仅对无历史新人启用画像

submit 下 CSV 与 ZIP 已同步更新。随包数据和原始 Excel 两条预测路径均复现同一 SHA-256 e4929d5abd3ffd663d716750c274bddbd28aee543835cf6814582238cf1a8e8a。610 位无历史新人使用画像，其他 346 位与原版逐行一致。此前提交材料备份在 outputs/submission_portrait_build/previous。

# 双塔挑战版已完成

见 DEEP_CHALLENGER.md，正式权重、挑战 CSV、回放与核验在 outputs/deep_challenger_v1。轻融合历史开发 7.492、复用诊断 4.785。冷书 537 个真值中选中版本前十未命中。保留当前 submit，未上传；暂无运行中的任务。

# 当前进度：官方匿名读者推荐与新馆藏实验已完成

建议提交 outputs/cold_demand_v1/提交结果_C题初赛.csv，956 人、9560 个位置已独立核验，未上传。完整历史回放 7.438/4.226（开发均分/复用诊断），模拟发放间隔 5.952/4.522。已运行画像、书目、BGE、遮蔽冷书、同系列、书单同伴及新人需求覆盖消融；未获得稳定超越原方案的证据。各入口、操作顺序、对照和后续切入点在 CATALOG_EXPERIMENT.md。暂无待恢复的训练进程。

# 老人组扩窗与预留位置实验已完成

11 个窗口、4 个评测日、5752 行独立核验已完成。六种备选中开发选定等权融合、零预留；扩窗学习提升未超过抽样波动。方法与复现见 OLDER_WINDOW_EXPERIMENT.md，结果在 outputs/older_call_windows_v1。

# 老人组索书号排序实验已完成

已完成七个快照、四个评价窗口以及 3595 行独立核验。完整方法、结果和复现命令见 OLDER_CALL_RANKER.md。默认模型保留原方案。

# 读者级别群体偏好：全部完成

更新时间：2026-11-01。代码、必要测试、正式对照、最终预测核验与使用说明均已完成，没有待恢复的训练任务。

## 结论

新增功能保留为可选项，默认关闭。三开发窗口综合分均下降：均值 0.062589247→0.061599032（-1.58%）。复用诊断综合分 0.060816120→0.057258946（-5.85%），差值区间 [-0.007873,-0.000280]。默认决定仅依据开发结果，未按诊断再次选择。

## 实现与口径

reader_level.py 保留原始级别标签；历史画像取截止前最近非缺失值，分组按事件时身份。组人数为不同读者数；每个级别内读者/图书对只取最大时间衰减贡献。门槛3人，平滑强度20，类别概率由条码概率汇总。小群体/未知组使用全局数值回退；AttributePrior 无有效级别组时保持原属性回退，有效组与旧先验按0.25/0.75混合。

HistoryContext、CandidateFeatures、AttributePrior 和 ranking.py 主入口已接通，独立读者 CSV 支持读者级别及画像时间检查。新增三个数值特征，没有新增分类特征或交叉组，故不需要分类-only实验。共享 ATTRIBUTES 未修改。配置、特征版本及源码哈希进入新协议和冻结计划；旧计划缺少开关时关闭。

## 验证与文件

- 51 项不同测试通过；最终预测改动后4项集成测试再次通过。
- 正式对照目录 outputs/reader_level_v2：8个历史快照×开关两组，4个评价窗口×两组，均已完成。训练250棵、推断前100棵、与optimized等权RRF，保持旧参数。
- 关闭组四窗口三指标精确复现。verify_reader_level.py 已独立重算8份推荐CSV，并检查8组快照、时间边界和新旧特征。
- 默认计划 outputs/reader_level_v2/locked_plan.json；可选开启 outputs/reader_level_v2/on/locked_plan.json。
- 全量默认结果 outputs/reader_level_prediction/result.csv，5112行×11列，覆盖所有历史目标读者，每人10个不同合法条码。
- 全量默认关闭方案与旧方案等价。reuse_reader_level_prediction.py 在比较8张训练特征表、来源和配置后复用已验证旧全历史模型/结果，未重复训练；reuse_provenance.json 记录来源及哈希。复制后通过最终独立核验。
- READER_LEVEL.md 包含公式、代码位置、配置、命令、完整新旧三指标、分组结果与限制；README 保持接口说明，DETAIL 提供扩展说明入口。
- 首次输出 outputs/reader_level 为失败尝试：直接训练100棵不等价于旧250棵的前100棵，精确复现保护已阻止其进入比较。正式结果只用v2。

## 使用

```powershell
python ranking.py predict
python ranking.py predict --plan outputs/reader_level_v2/on/locked_plan.json --readers readers.csv --output-dir outputs/reader_level_on_prediction
python verify_reader_level.py --output-dir outputs/reader_level_v2 --prediction-dir outputs/reader_level_prediction
```

predict 默认读取已完成的v2选定计划，重新训练并生成结果；现有默认结果已完成，不必重复。新特征/参数实验须另选输出目录，保留旧数据和全部报告。

运行环境仍为项目 .model_deps 与此前 Python 3.12.14 / NumPy 2.3.5 / pandas 3.0.1 / CatBoost 1.2.8。Python 位于 C:/Users/Cyanwave/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe；使用 -X utf8 和绝对路径。不设置 BLAS 线程环境变量，以保持旧结果复现。

后续优化分析仍可参考 SCORE_DIAGNOSIS.md。本次功能完成不代表取得性能提升，也未验证真实独立冷读者画像的收益。

## 借阅事件模型：四轮与八轮的续跑结果

三个开发窗已完成同轨迹四/八轮对照，事件轻融合 7.194→8.085；复用诊断 5.864→5.507。正式提交未改。详细分组、采样误差、结果路径与复核步骤见 [EVENT_SEQUENCE_RESULTS.md](EVENT_SEQUENCE_RESULTS.md)。

## 实验 C：下一事件、整窗与联合监督

已完成两种子 × 三个目标 × 三个开发窗，按主实验选联合，复用诊断联合 5.422、整窗 4.401。八轮轻融合两种子开发均分：下一事件 7.813、整窗 7.845、联合 7.933。下一步 B 可固定联合目标与同等更新预算。逐窗指标、复核命令及输出见 [实验 C 报告](EVENT_OBJECTIVE_C_RESULTS.md)。

## 实验 B · 读者历史编码

固定联合目标与八轮预算，三个开发窗和两个种子的正式书单轻融合均分：逐本加权 7.886、事件加权 7.799、事件 GRU 7.933。复用诊断依次为 5.146、5.033、5.076。GRU 与逐本加权的差值落在读者抽样波动范围内。见 [实验 B 报告](EVENT_ENCODER_B_RESULTS.md)。

## 轻量 SASRec 对照

两个种子、三开发窗和复用诊断已全部完成。采用现提交分流的 GRU-8 和 SASRec-32 开发均分为 7.834 和 7.666。仅长书单换 SASRec-32 的整表差值为开发 +0.199、复用诊断 −0.148。建议保留 GRU-8。见 [实验报告](SASREC_LENGTH_RESULTS.md)。
