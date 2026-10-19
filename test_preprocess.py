"""检验会改变推荐数据含义的边界：编号、重复借阅、日期及缺失处理。"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from preprocess import clean_frame, parse_dates, publication_year, save_csv


def row(**updates):
    data = {"序号": "1", "操作类型": "借书", "条形码": "B0001", "登录号": "A0001",
            "读者条码": "001234567890123456", "处理时间": "2022-03-01 09:00:00",
            "单位": "文学院", "读者级别": "学生", "题名": "示例图书", "责任者": "作者",
            "出版者": "出版社", "出版日期": "2021.05", "馆藏地址": "文科图书室1",
            "索书号": "I247.5/1001", "年级": "2020级", "班级": "2020级汉语言文学",
            "专业": "汉语言文学", "系别": "文学院", "超期天数": "0", "赔罚款": "0",
            "外借时间": "", "应归还时间": "", "流通库室键码": "", "附件": "", "架位号": ""}
    data.update(updates)
    return data


class PreprocessTests(unittest.TestCase):
    def clean(self, *rows):
        return clean_frame(pd.DataFrame(rows))

    def test_identifiers_and_csv_keep_leading_zeros_and_long_text(self):
        result = self.clean(row(条形码="00001234"))["cleaned"]
        base = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(prefix="preprocess_test_", dir=base) as folder:
            self.assertEqual(Path(folder).resolve().parent, base)
            p = Path(folder) / "clean.csv"
            save_csv(result, p)
            loaded = pd.read_csv(p, dtype="string", keep_default_na=False, encoding="utf-8-sig")
        self.assertEqual(loaded.loc[0, "读者条码"], "001234567890123456")
        self.assertEqual(loaded.loc[0, "条形码"], "00001234")

    def test_only_exact_business_duplicates_are_removed(self):
        result = self.clean(row(), row(序号="2"), row(序号="3", 处理时间="2022-03-02 09:00:00"))
        self.assertEqual(len(result["cleaned"]), 2)
        self.assertEqual(result["summary"]["duplicate_rows_removed"], 1)
        self.assertEqual(result["summary"]["after"]["repeated_user_book_events"], 1)
        self.assertTrue(result["summary"]["row_reconciliation_passed"])

    def test_missing_keys_bad_time_and_non_borrow_are_rejected(self):
        result = self.clean(row(), row(读者条码=""), row(条形码=""), row(处理时间="日期错误"), row(操作类型="还书"))
        self.assertEqual(len(result["cleaned"]), 1)
        self.assertEqual(result["summary"]["rejected_rows"], 4)

    def test_same_event_different_metadata_is_not_silently_overwritten(self):
        result = self.clean(row(), row(序号="2", 题名="另一个题名"))
        self.assertEqual(len(result["cleaned"]), 2)
        self.assertTrue(result["cleaned"]["quality_flags"].str.contains("event_key_conflict").all())
        self.assertGreater(result["summary"]["checks"]["entity_conflict_条形码_题名"], 0)

    def test_optional_missing_does_not_fake_factual_completeness(self):
        result = self.clean(row(专业="", 责任者=""))
        cleaned = result["cleaned"].iloc[0]
        self.assertEqual(cleaned["专业"], "未知")
        quality = result["column_quality"].set_index("column")
        self.assertEqual(quality.loc["专业", "clean_effective_missing"], 1)
        self.assertEqual(quality.loc["专业", "clean_null_cells"], 0)

    def test_grade_recovered_only_from_current_row_and_not_future_cohort(self):
        result = self.clean(row(年级="图书馆"), row(序号="2", 读者条码="U2", 年级="2024级", 班级="2024级英语"))
        cleaned = result["cleaned"].set_index("读者条码")
        self.assertEqual(cleaned.loc["001234567890123456", "年级"], 2020)
        self.assertTrue(pd.isna(cleaned.loc["U2", "年级"]))

    def test_later_rows_do_not_fill_earlier_metadata(self):
        result = self.clean(row(责任者="", 专业=""), row(序号="2", 处理时间="2023-03-01 09:00:00"))
        earlier = result["cleaned"].iloc[0]
        self.assertEqual(earlier["责任者"], "未知")
        self.assertEqual(earlier["专业"], "未知")

    def test_publication_year_mixed_precision_and_ambiguity(self):
        for raw, expected in [("202005", 2020), ("[2004]", 2004), ("2005重印", 2005), ("c1999.", 1999), ("2020.12", 2020)]:
            self.assertEqual(publication_year(raw)[0], expected)
        self.assertEqual(publication_year("2000 d2005重印")[1], "ambiguous")
        self.assertEqual(publication_year("30cm")[1], "invalid")
        self.assertEqual(publication_year("202013")[1], "invalid")

    def test_invalid_numeric_stays_missing_and_zero_is_kept(self):
        result = self.clean(row(超期天数="-1", 赔罚款="inf"), row(序号="2", 处理时间="2022-03-02", 超期天数="1.5"), row(序号="3", 处理时间="2022-03-03"))
        cleaned = result["cleaned"].sort_values("source_excel_row")
        self.assertTrue(pd.isna(cleaned.iloc[0]["超期天数"]))
        self.assertTrue(pd.isna(cleaned.iloc[0]["赔罚款"]))
        self.assertTrue(pd.isna(cleaned.iloc[1]["超期天数"]))
        self.assertEqual(cleaned.iloc[2]["超期天数"], 0)

    def test_loan_time_mismatch_is_retained_and_missing_not_imputed(self):
        result = self.clean(row(外借时间="2023-03-01"), row(序号="2", 处理时间="2022-03-02"))
        cleaned = result["cleaned"].sort_values("source_excel_row")
        self.assertIn("loan_time_after_event", cleaned.iloc[0]["quality_flags"])
        self.assertEqual(cleaned.iloc[0]["处理时间"], pd.Timestamp("2022-03-01 09:00:00"))
        self.assertTrue(pd.isna(cleaned.iloc[1]["外借时间"]))
        parsed = parse_dates(pd.Series(["44562", "2022-01-01 08:00:00", "错误"]))
        self.assertEqual(parsed.iloc[0], pd.Timestamp("2022-01-01"))
        self.assertTrue(pd.isna(parsed.iloc[2]))

    def test_sparse_matrix_uses_unique_pairs_not_repeated_events(self):
        result = self.clean(row(), row(序号="2", 处理时间="2022-03-02"))
        self.assertEqual(result["summary"]["after"]["interaction_density"], 1.0)

    def test_empty_input_keeps_output_schema(self):
        result = clean_frame(pd.DataFrame(columns=list(row())))
        self.assertEqual(len(result["cleaned"]), 0)
        self.assertIsNone(result["summary"]["retention_rate"])
        self.assertTrue(result["summary"]["row_reconciliation_passed"])

    def test_all_records_rejected_still_produces_auditable_result(self):
        result = self.clean(row(读者条码=""), row(处理时间="bad"))
        self.assertEqual(len(result["cleaned"]), 0)
        self.assertEqual(len(result["removed"]), 2)
        self.assertEqual(result["summary"]["rejected_rows"], 2)


if __name__ == "__main__":
    unittest.main()
