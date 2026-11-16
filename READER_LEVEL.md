# 读者级别群体偏好

本功能使用原始字段 `读者级别`，不合并学生、本科学生、教师、教职工等不同标签，也不增加交叉分组。无效值沿用 `attribute_value`。本轮只增加数值偏好，不增加级别分类特征。

## 计算口径

截止时刻为 t，半衰期 H 沿用原模型（默认 180 天）。对级别 g 内读者 u 与条码 i 的历史事件：

`w(g,u,i) = max[2^(-(t-event_time)/H)]`

最大值仅在事件时级别为 g 的记录中计算。同一读者重复借阅同一册不会按次数累加，但最近一次借阅可以刷新衰减权重。不同图书仍分别贡献。级别群体支持人数是该组不同读者数，不是事件数或读者—图书对数；默认至少 3 人。

全局先按读者—条码对做同样的最大衰减去重，再归一化得到 P0(i)。合格群体的计数 Cg(i)=sum_u w(g,u,i)，图书概率为：

`Pg(i) = [Cg(i) + 20 * P0(i)] / [sum_j Cg(j) + 20]`

类别沿用原排序模型的索书号分类前缀，类别概率为该前缀所有历史条码的 Pg 之和。类别贡献因此同样受到读者—条码去重控制，但不限制一个读者对不同书的正常贡献。

级别缺失、未见或人数不足时，数值特征回退到全局分布，`level_group_available=0`。历史当前级别取截止前最后一次非缺失值；历史统计使用各次借阅当时的级别。身份变化不会把早期行为重新归入新身份。

## 接入位置和兼容性

- `reader_level.py`：`ReaderLevelConfig`、`ReaderLevelPreference`，默认关闭。
- `advanced_models.py`：`HistoryContext(..., reader_level=True)` 可选补充历史级别画像；共享 `ATTRIBUTES` 保持不变。
- `ranking_models.py`：`CandidateFeatures(..., reader_level_config=...)` 增加 `level_item_probability`、`level_category_probability`、`level_group_available`。均为数值特征，CatBoost 类别列声明不变；模型保存的列清单包含新增特征。
- `optimized_models.py`：`AttributePrior(..., reader_level_config=...)` 对合格级别采用 `0.75 * 原属性先验 + 0.25 * 级别先验`。没有合格级别时原先验原样返回，包括原有专业/系别/学习阶段回退。
- `ranking.py`：评估开关保存到冻结计划，预测从计划读取；独立读者 CSV 的级别同时进入候选特征和新读者先验。可选 `画像时间` 必须早于预测时点。读者级别未知时不得从目标期事件回填。
- `reader_level_experiment.py`：固定第三轮的扩展候选、100 棵树、等权 RRF 结构，比较开/关。每个预测窗口只用更早结束的标签窗口训练。

本轮不新增召回通道。两臂复用第三轮不可变的候选及旧特征输入，在独立目录重建派生特征文件，并重新训练两组排序器；原列逐项断言相等。关闭臂三开发窗口和诊断分数必须精确复现原报告，否则停止比较。不是修改旧缓存身份使其冒充新版。新协议包含来源身份、配置、特征模块及依赖源码哈希，配置变化必须换输出目录。

## 命令

在本目录及原 Python 数值环境中执行，不更改 BLAS 线程环境变量。

```powershell
# 两臂固定参数对照。参数在运行前固定，不读取诊断成绩调参。
python reader_level_experiment.py --output-dir outputs/reader_level_v2
# 新选定方案（包含已冻结开关）全历史预测
python ranking.py predict --plan outputs/reader_level_v2/locked_plan.json --output-dir outputs/reader_level_prediction
# 强制使用可选开启方案；选择对应冻结计划，不在预测时随意改训练开关
python ranking.py predict --plan outputs/reader_level_v2/on/locked_plan.json --readers readers.csv --output-dir outputs/reader_level_on_prediction
# 如需完整重建全部召回及比较原18种融合结构，须另用目录
python ranking.py evaluate --reader-level --level-min-users 3 --level-strength 20 --level-prior-weight 0.25 --output-dir outputs/ranking_level_full
python -m unittest test_reader_level test_reader_level_integration test_ranking test_optimized test_advanced
```

读者清单必需列是 `读者条码`，可附 `读者级别`、专业、系别、年级、画像时间。标签不做语义归并，必须与历史标签匹配才会使用对应组。旧冻结计划没有 `reader_level` 字段时默认关闭，旧冻结计划仍可用于预测；新源码重新评估必须选择新输出目录，不能覆盖旧实验缓存。

## 实验与测试状态

2026-10-31：9 项新增基础及集成测试、42 项受影响既有测试，共 **51 项通过**。包含原始标签不合并、缺失和小群体回退、重复控制、历史身份变化、未来事件拒绝、独立新读者、关闭时特征及先验一致、真实 CatBoost 保存加载、读者 CSV 预测接口及画像时间拒绝。

正式实验已完成。输出为 `outputs/reader_level_v2/`，`protocol.json` 记录预设协议，`off/` 与 `on/` 保存独立特征、模型、预测、历史身份分组结果。开发均分决定选定开关（平分选关闭），冻结后才评价复用诊断集。实测未获得收益，默认关闭，详见下文。

训练严格沿用原方案的 250 棵树，再截取前 100 棵进行推断。首次试图直接训练 100 棵未通过关闭臂复现检查，相关输出留在 outputs/reader_level，仅供失败复核，不纳入正式比较。修正实验目录为 outputs/reader_level_v2。

## 正式结果与交付状态（2026-11-01）

正式开关对照已全部完成，关闭臂的三个开发窗口和复用诊断窗口均精确复现旧方案。结果如下；2024-11-01 起的最后一组是复用诊断，不是独立盲测。

| 窗口 | 开关 | Recall@10 | NDCG@10 | 综合分 |
| --- | --- | ---: | ---: | ---: |
| 2023-09-01 | 关闭 | 0.062084 | 0.051976 | 0.058041 |
| 2023-09-01 | 开启 | 0.061535 | 0.051589 | 0.057556 |
| 2024-03-01 | 关闭 | 0.083881 | 0.072334 | 0.079262 |
| 2024-03-01 | 开启 | 0.082363 | 0.071717 | 0.078105 |
| 2024-09-01 | 关闭 | 0.054570 | 0.044307 | 0.050465 |
| 2024-09-01 | 开启 | 0.053162 | 0.043097 | 0.049136 |
| 2024-11-01 | 关闭 | 0.064349 | 0.055517 | 0.060816 |
| 2024-11-01 | 开启 | 0.059583 | 0.053773 | 0.057259 |


三开发窗口综合分均值：关闭 **0.062589247**，开启 **0.061599032**，相对变化 **-1.58%**。三个开发窗口均下降，因此在读取本轮诊断成绩前已冻结 **默认关闭**。新增功能及开启计划保留，未根据诊断结果再调参数。

复用诊断综合分相对变化 **-5.85%**；配对 bootstrap 5000 次的差值 95% 区间为 **[-0.007873, -0.000280]**。7 位读者改善、13 位变差、569 位持平。这支持当前配置未带来收益，不代表所有读者级别方法都无效，也不能把复用诊断的区间视为正式比赛保证。

诊断分组中，教师综合分 0.152064→0.152700、社会读者 0.180098→0.181042；学生 0.116437→0.104063、本科学生 0.154174→0.141265。未知身份的 335 位读者保持 0.002716。各窗口完整身份分组含人数、Recall、NDCG、综合分，保存在 experiment_report.json 的 by_level 中；小组只作描述，不据此再选择子组开关。

人数门槛 3、平滑强度 20、属性先验混合权重 0.25 均在实验前固定，未按诊断调参。离线冷读者没有独立画像，因此本实验未验证真实冷读者画像回退的收益，也未证明 0.25 是最优混合权重；该路径已通过独立画像行为及端到端测试。

## 最终核验与产物

- 51 项不同测试全部通过；最后加入预测版本检查后，4 项集成测试再次通过。未把重复运行计成新增测试数。
- verify_reader_level.py 独立核验 8 组开关快照、重算 8 份评价推荐 CSV 的指标；确认标签时间边界、旧特征一致及候选合法。
- 最终预测 **5112 位读者 × 10 个条码**，所有目标历史读者均覆盖，每行无重复且条码均属于截止前馆藏。
- 默认关闭与原全历史方案等价：reuse_reader_level_prediction.py 对全部 8 张训练特征表逐项比较、核对配置和来源后，复用原全历史模型与推荐，未重复训练或冒称重新计算。来源与文件 SHA256 记录在 outputs/reader_level_prediction/reuse_provenance.json，复制后的文件已再次独立核验。
- outputs/reader_level_v2/experiment_report.json、comparison.csv：正式实验；off/ 和 on/：两组重训模型、特征、推荐及分组报告。
- outputs/reader_level_v2/locked_plan.json：默认关闭计划；on/locked_plan.json：可选开启计划。
- outputs/reader_level_prediction/result.csv：最终默认结果；prediction_summary.json 明确记录复用来源与开关。
- outputs/reader_level_v2/verification_summary.json：最终独立核验结果。

完整报告存在后，python ranking.py predict 默认读取本轮选定计划，写入 outputs/reader_level_prediction，保留原 outputs/ranking_prediction。强制开启请显式传入 on/locked_plan.json。若希望重新训练全历史默认模型，可运行前述 predict 命令；目前等价默认结果已经核验完毕，无须为获得相同结果重复计算。

```powershell
python verify_reader_level.py --output-dir outputs/reader_level_v2 --prediction-dir outputs/reader_level_prediction
```
