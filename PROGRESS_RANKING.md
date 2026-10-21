# 第三轮优化进度与恢复说明

更新时间：2026-10-21。**本轮任务已全部完成**：内容候选扩展、学习排序、旧模型对比、全历史预测及独立核验均已完成。此前预测进程已正常结束，无需恢复训练。

## 冻结方案与成绩

最终方案为 `expanded_100_blend1`：六路既有候选各前 200，增加 BGE 中文语义及类别候选各前 100；CatBoost 使用前 100 棵树排序，与第二轮 optimized 按 1:1 的 RRF 权重融合。专业、系别、学习阶段已经进入特征，但本轮没有单独移除属性的对照，不能把总增益归因于某个属性。

| 方法 | 诊断 Recall@10 | 诊断 NDCG@10 | 诊断综合分 |
| --- | ---: | ---: | ---: |
| 原 hybrid 基线 | 0.0541896742 | 0.0407195555 | 0.0488016267 |
| 第一轮 previous | 0.0593604578 | 0.0498688982 | 0.0555638340 |
| recent | 0.0602850790 | 0.0540267570 | 0.0577817502 |
| 第二轮 optimized | 0.0612547827 | 0.0554866475 | 0.0589475286 |
| 第三轮选定方案 | 0.0643488059 | 0.0555170914 | 0.0608161201 |

三个开发窗口的第二轮得分为 0.0560047152、0.0789616082、0.0483751388，均值 0.0611138207；第三轮为 0.0580410821、0.0792621325、0.0504645277，均值 0.0625892474，相对提高 2.41%。

诊断综合分相对第二轮提高 3.17%，相对原基线累计提高 24.62%。但相对第二轮的配对 bootstrap 95% 差值区间 **[-0.0019281525, +0.0063002829]** 包含零，只能称小幅改善，不能称显著领先。15 位读者改善、10 位变差、564 位持平。旧诊断集反复使用，不是盲测。

18 种预声明方案仅用三个开发窗口的等权平均综合分选择；冻结后评价诊断，没有按诊断成绩改方案。

## 已完成产物与验证

- 新增 `ranking_models.py`、`ranking.py`、`requirements-ranking.txt`、`test_ranking.py`、`verify_ranking.py`。
- 11 项测试已通过，包含真实 CatBoost 训练、保存加载、时间隔离、正例不注入和冷读者回退。
- `DETAIL.md` 已补齐方法、成绩、置信区间、候选覆盖、命中分解、限制与复现记录；README 只补接口。
- `outputs/ranking/experiment_report.json`：完整正式报告；`comparison.csv`：诊断对比；`trials.json`：全部开发方案。
- `outputs/ranking/locked_plan.json`：冻结结构；`snapshots/`：7 个训练快照及 1 个诊断快照的特征、标签、模型与推荐。
- `outputs/ranking/verification_summary.json`：独立重算 **70 份评价 CSV**，核验 **8 个快照**，四个旧参照三项指标精确复现，原 Excel 未变化。
- `outputs/ranking_prediction/result.csv`：已生成并通过核验，**5112 行 × 11 列**，每位读者 10 个不重复的合法历史条码。
- 同目录的 `prediction_summary.json`、`ranker.cbm`、`ranker.features.json`、`ranker_training.json` 保存来源、结构、模型及最终训练窗口。

全历史预测时点为 `2025-01-16T11:20:27.000000001`，使用 8 个完整窗口、942 个有效排序组和 2022 个候选正例。原诊断窗口此时仅用于最终训练，不重新选型、不改写离线成绩。正式提交仍需按赛方的截止时点、目标读者及模板配置。

## 瓶颈和后续方向

- 已知馆藏未借真值的候选覆盖从 130/575 增至 137/575，最终前十命中由 5 增至 7，探索增益有限。
- 复借命中从 98 降至 92，总命中从 103 降至 99；宏平均综合分上升不等于总命中增加，必须保持比赛评价口径。
- 967 对未知馆藏真值不可达；335 位无历史读者成绩未改善，仍使用旧模型回退。
- 直接固定融合新召回、纯排序器都不如保留旧方案的排序融合。
- 下一轮优先补充预测时已知的独立馆藏和读者画像；当前数据内可研究书目候选及复本条码分配，但先固定新的开发规则，避免继续围绕同一诊断集调参。
- **没有遗留必须补跑的任务。**新实验使用新输出目录，保留本轮冻结结果作参照。

## 环境与复现

- 项目：`E:\match\大三上\scdmc\c\code`。
- Python：`C:\Users\Cyanwave\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`，加 `-X utf8`。
- 本地依赖在 `.model_deps`；受限执行环境加载 Torch/SciPy/CatBoost 原生 DLL 需提升执行权限。使用绝对路径，避免默认工作目录偏移。
- 不设置 `OPENBLAS_NUM_THREADS`、`OMP_NUM_THREADS`、`MKL_NUM_THREADS`；它们会影响旧 Item2Vec 近似并列分值的排序。
- 环境：Python 3.12.14、NumPy 2.3.5、pandas 3.0.1、SciPy 1.18.1、PyTorch 2.14.1+cpu、CatBoost 1.2.8、transformers 4.57.6；完整记录见 `outputs/ranking/environment.json`。
- BGE：`BAAI/bge-small-zh-v1.5`，revision `7999e1d3359715c523056ef9478215996d62a620`；权重 SHA256 `354763b9b1357bc9c44f62c6be2276321081ed2567773608c0d0785b61d5a026`。官方下载与校验已完成，无需重新下载。
- 原 Excel SHA256：`3a7da842f9593ef0cbc23f283c0c33babc246b6a678cb06f3717435d483a829d`。原模型、Excel、旧报告和缓存均保留。

需要复现时，在项目目录的对应 Python 环境运行：

```powershell
python ranking.py evaluate
python ranking.py predict
python verify_ranking.py --prediction-dir outputs/ranking_prediction
```

以上都已实际执行完成。`evaluate` 复用身份匹配的完整快照及已完成比较；`predict` 会重新计算全量推荐，已有结果正常时无需重复运行。改动特征源码或数据后请另选输出目录，避免混用缓存。
