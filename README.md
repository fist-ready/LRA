# LRA 图书馆推荐算法

GitHub 源码版：根目录是清洗、建模、实验和测试代码；[submission/](submission/) 是正式提交包中的训练、预测和校验源码，见[建模流程](submission/建模过程.md)。借阅数据、验证名单、主题标注结果、模型权重、虚拟环境和实验缓存仅在本地保存。源码版不含完整比赛输入，无法仅凭克隆直接生成正式预测。新建源码子目录时，请在 `.gitignore` 中显式放行需要发布的文件。

## 读者级别群体偏好接口

```powershell
python reader_level_experiment.py --output-dir outputs/reader_level_v2
python ranking.py predict --plan outputs/reader_level_v2/locked_plan.json --output-dir outputs/reader_level_prediction
python verify_reader_level.py --output-dir outputs/reader_level_v2 --prediction-dir outputs/reader_level_prediction
```

独立读者清单可增加 `读者级别` 列。`ranking.py evaluate --reader-level` 启用群体数值特征；可配置 `--level-min-users`、`--level-strength`、`--level-prior-weight`，须使用新输出目录。预测开关由冻结计划决定，旧计划缺省关闭。公式、回退、对照和测试说明见 [READER_LEVEL.md](READER_LEVEL.md)。

本轮完整报告存在时，`ranking.py predict` 默认读取 `outputs/reader_level_v2/locked_plan.json`，输出到 `outputs/reader_level_prediction/`；当前选定关闭。可选开启使用 `--plan outputs/reader_level_v2/on/locked_plan.json`，建议另指定输出目录。

## 数据预处理模块

### 用法

```powershell
python preprocess.py
```

需要指定输入、输出目录或工作表时：

```powershell
python preprocess.py --input "input path" --output-dir "output directory" --sheet "work sheet name"
```

- `--input`：输入 Excel 文件，可使用绝对路径或相对路径。默认值：脚本目录下的 `data/2022-2025处理后借阅信息.xlsx`。

- `--output-dir`：输出目录，脚本在其中保存三个 CSV 文件。默认值：脚本目录下的 `data/processed`。

- `--sheet`：Excel 底部标签页的名称；省略时自动定位借阅明细工作表。默认值：自动识别；当前数据为 `Sheet1`。

### 输出

- `data/processed/borrowing_cleaned.csv`：清洗后的借阅明细、派生属性和问题标记。用途：后续特征处理与推荐建模。

- `data/processed/quality_metrics.csv`：数据规模、完整率、重复量、稀疏度、检查命中数及处理结果。用途：判断数据整体情况及清洗影响。

- `data/processed/field_quality.csv`：每个原始字段的缺失数量、缺失率和“未知”数量。用途：定位需要处理或谨慎使用的字段。

## 推荐基线接口

在本目录运行，无需新增深度学习框架：

```powershell
python -m pip install -r requirements.txt
python baseline.py evaluate
python baseline.py predict
```

`evaluate` 按时间划分训练、验证、本地测试集，比较 `popular`、`itemcf`、`hybrid`，在验证集上选择方法。`predict` 默认读取验证报告，用全部公开历史重新训练，为所有历史读者生成 Top-10。

| 参数 | 接口说明 |
| --- | --- |
| `--input`、`--output-dir`、`--sheet` | 两个命令通用；输入支持原 Excel 或清洗 CSV；默认输入与预处理模块相同 |
| `evaluate --valid-start`、`--test-start`、`--test-end` | 默认验证起点 `2024-09-01`、测试起点 `2024-11-01`；测试排他上界默认不限 |
| `--neighbors`、`--max-history`、`--half-life-days`、`--shrinkage`、`--group-min-users` | 默认依次为 `50`、`100`、`180`、`10`、`5` |
| `--exclude-seen`、`--k` | 可排除历史已借图书；默认允许复借、`k=10` |
| `predict --method`、`--selection-report` | 默认 `auto` 读取 `outputs/baseline/evaluation_report.json` 的方法与参数，也可显式选三种方法之一 |
| `predict --readers`、`--as-of` | 可指定读者 CSV（必需列：读者条码，可附专业/系别/年级），以及训练记录的排他时间上界 |

默认评估输出为 `outputs/baseline/` 下的 `splits/`、`metrics.csv`、`evaluation_report.json`、各方法推荐与逐读者指标。默认预测输出为 `outputs/prediction/result.csv` 及 `prediction_summary.json`。

`result.csv` 暂用 `读者条码,推荐1,...,推荐10` 的本地格式，正式模板发布后再适配。原 Excel 保持只读。参数详情用 `python baseline.py evaluate --help` / `python baseline.py predict --help` 查看；原理、指标和实验结果见 [DETAIL.md](DETAIL.md)。

```powershell
python test_baseline.py
```

## 多路推荐实验接口

```powershell
python -m pip install -r requirements-advanced.txt
python advanced.py evaluate
python advanced.py predict
python test_advanced.py
```

`evaluate` 比较原融合基线、BPR-MF、属性 BPR、Item2Vec、GRU、内容及最近借阅分支，并仅在验证集选择轮数和融合权重。`predict` 读取冻结计划，用全历史重新训练选定分支。

- 通用：`--input`、`--output-dir`、`--sheet`。
- 评估：`--valid-start`、`--test-start`、`--test-end` 沿用基线；`--dimensions 32`、`--mf-epochs 30`、`--item2vec-epochs 12`、`--gru-epochs 24`、`--candidate-k 200`、`--seed 2026`、`--threads 4` 可调整。其他选项见 `evaluate --help`。
- 预测：`--plan` 默认 `outputs/advanced/locked_plan.json`；`--readers` 当前仅读取读者条码；`--as-of` 可限制历史排他上界。
- 默认评估输出：`outputs/advanced/experiment_report.json`、`metrics.csv`、验证试验表、`locked_plan.json`、各方法 Top-10、逐读者指标及模型权重；配对重采样区间位于报告的 `paired_comparison`。
- 默认预测输出：`outputs/advanced_prediction/result.csv` 及训练摘要。原基线输出保持在原目录，实验原理和结果见 [DETAIL.md](DETAIL.md)。

## 多时间窗口优化接口

先运行上一节的 `advanced.py evaluate` 生成参照计划；同一组对比使用相同的 Python 与数值库环境。

```powershell
python -m pip install -r requirements-optimized.txt
python optimized.py evaluate
python optimized.py predict
python test_optimized.py
```

`evaluate` 用三个历史窗口选择 EASE、RP3、书目聚合、属性先验及复借探索的参数和组合，再与原基线、上一轮融合、最近借阅策略比较。`predict` 读取本轮冻结计划，用全部公开历史生成 Top-10。

- 通用：`--input`、`--output-dir`、`--sheet`。
- 评估：`--previous-plan` 默认上一轮的 `outputs/advanced/locked_plan.json`；`--dev-window START END` 可重复指定开发窗口；`--diagnostic-start` 默认 `2024-11-01`，`--diagnostic-end` 为可选排他终点；`--bootstrap-samples` 默认 `5000`。
- 重算已冻结实验可加 `evaluate --reuse-selection PLAN_PATH`，保持该计划的参数和组合。
- 预测：`--plan` 默认 `outputs/optimized/locked_plan.json`；`--readers` 可附预测前独立已知的专业、系别、年级；可选 `画像时间` 必须严格早于 `--as-of`。省略读者清单时为全部历史读者推荐。
- 默认输出：`outputs/optimized/` 保存协议、参数试验、策略试验、冻结计划、各窗口推荐、逐读者指标与 `experiment_report.json`；全历史预测在 `outputs/optimized_prediction/result.csv`。

旧测试区间已经看过，本轮将其称为“复用诊断区间”。算法、验证口径、属性消融及完整结果见 [DETAIL.md](DETAIL.md)。

## 内容召回与学习排序接口

需要上一节生成的冻结计划和参照报告。

```powershell
python -m pip install -r requirements-ranking.txt
python ranking.py prepare-model
python test_ranking.py
python ranking.py evaluate
python ranking.py predict
python verify_ranking.py --prediction-dir outputs/ranking_prediction
```

- `prepare-model --model-dir`：下载并记录官方 BGE 中文小模型版本与校验值；默认放入 `.model_deps/bge-small-zh-v1.5/`。
- `evaluate --previous-plan --model-dir --input --output-dir`：默认沿用第二轮冻结方案；比较现有召回排序、扩展内容召回与扩展排序。中断后运行相同命令，自动复用完整快照与已完成比较。
- `predict --plan --input --readers --as-of --output-dir`：默认使用 `outputs/ranking/locked_plan.json`；读者 CSV 可附预测前独立已知的专业、系别、年级和画像时间。
- 评估输出：`outputs/ranking/experiment_report.json`、`comparison.csv`、`locked_plan.json`、`progress.json`、历史快照特征、推荐与排序权重。
- 全历史预测：`outputs/ranking_prediction/result.csv`。方法与结果见 [DETAIL.md](DETAIL.md)，中断恢复记录见 [PROGRESS_RANKING.md](PROGRESS_RANKING.md)。
- `verify_ranking.py --output-dir --prediction-dir`：从导出列表重算评价指标并核验候选和时间边界；`--prediction-dir` 可选，默认审计目录为 `outputs/ranking/`。


老人组索书号排序回放：`python older_call_ranker.py`；独立核验：`python verify_older_call_ranker.py`。详见 [OLDER_CALL_RANKER.md](OLDER_CALL_RANKER.md)。

老人组扩窗与名额回放：依次运行 `python prepare_older_windows.py`、`python evaluate_older_windows.py`。详见 [OLDER_WINDOW_EXPERIMENT.md](OLDER_WINDOW_EXPERIMENT.md)。

新人组分流：`python newcomer_replay.py`，复核：`python verify_newcomer.py`。见 [NEWCOMER_REPLAY.md](NEWCOMER_REPLAY.md)。

官方匿名读者与新入藏目录流程：入口 `catalog_experiment.py`，最终提交校验 `verify_contest_submission.py`，完整运行顺序见 [CATALOG_EXPERIMENT.md](CATALOG_EXPERIMENT.md)。

双塔挑战版：`python deep_challenger.py` 训练与回放；`python deep_challenger.py --predict-only` 加载权重预测。详见 [DEEP_CHALLENGER.md](DEEP_CHALLENGER.md)。

新人多属性画像挑战、回放与导出见 [NEWCOMER_PORTRAIT.md](NEWCOMER_PORTRAIT.md)。

主题标签与借阅事件时序模型训练：`powershell -NoProfile -ExecutionPolicy Bypass -File .\run_event_training.ps1 --suite`。输入、权重、对照与恢复见 [EVENT_SEQUENCE_TRAINING.md](EVENT_SEQUENCE_TRAINING.md)。

自选轮数的三窗评分表：`powershell -NoProfile -ExecutionPolicy Bypass -File .\run_event_training.ps1 --epoch-study --epochs 4 8 12`。可只填一个正整数，也可空格分隔多个；按每窗同一条训练轨迹取各指定轮，两个种子等权，再取三窗均分。完成后直接打印表格，并保存至 `outputs/custom_epoch_study/<轮数组合>/epoch_table.md` 和同名 CSV。重复同一组合时复用已完成的窗口和种子，加 `--fresh` 可重训。仅运行 `--epoch-study` 可立即查看原有 8、12、16 轮表。
