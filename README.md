# LCA 图书馆推荐算法

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


