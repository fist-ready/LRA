"""借阅数据预处理：读取 Excel、清洗数据、保存数据及两份质量指标 CSV。"""

import argparse
import re
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

CORE = ["读者条码", "条形码", "处理时间", "操作类型"]
EVENT_KEY = ["读者条码", "条形码", "处理时间"]
SOURCE = ["source_sheet", "source_excel_row"]
UNKNOWN = "未知"
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
TEXT_COLUMNS = ["单位", "读者级别", "题名", "责任者", "出版者", "馆藏地址", "索书号", "班级", "专业", "系别"]


def text(value):
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and value.is_integer():
        return str(int(value))
    return value.isoformat(sep=" ") if isinstance(value, (datetime, pd.Timestamp)) else str(value)


def normalize(value, column):
    value = unicodedata.normalize("NFC", text(value)).replace("\u200b", "").replace("\ufeff", "")
    value = re.sub(r"\s+", " ", CONTROL.sub(" ", value)).strip()
    missing = value.casefold() in {"", "null", "none", "nan", "n/a", "<na>"}
    if column not in ["读者条码", "条形码", "登录号"]:
        missing |= value in {UNKNOWN, "不详", "未填写", "[出版年不详]", "[出版日期不详]"}
    return pd.NA if missing or (column == "单位" and value == "单位") else value


def read_source(path, sheet=None):
    """识别真实表头，本数据的表头在第 3 行。编号按原文本读取。"""
    with pd.ExcelFile(path, engine="openpyxl") as book:
        matches = []
        for name in ([sheet] if sheet else book.sheet_names):
            preview = pd.read_excel(book, sheet_name=name, header=None, nrows=20, dtype=object, keep_default_na=False)
            for i, row in preview.iterrows():
                headers = [text(v).strip() for v in row]
                if set(CORE).issubset(headers):
                    if len([h for h in headers if h]) != len(set(h for h in headers if h)):
                        raise ValueError("表头存在重复字段，请核对源文件。")
                    matches.append((name, int(i)))
                    break
        if len(matches) != 1:
            raise ValueError("无法唯一定位借阅明细表，请用 --sheet 指定工作表。")
        name, header = matches[0]
        data = pd.read_excel(book, sheet_name=name, header=header, dtype=object, keep_default_na=False)
    data.columns = [text(c).strip() for c in data.columns]
    if set(SOURCE) & set(data.columns):
        raise ValueError("源表含预留的溯源字段名。")
    data["source_sheet"], data["source_excel_row"] = name, np.arange(len(data)) + header + 2
    return data


def parse_dates(series):
    values = series.astype("string")
    serial = values.str.fullmatch(r"\d{1,5}(?:\.\d+)?").fillna(False)
    numbers = pd.to_numeric(values.where(serial), errors="coerce")
    serial &= numbers.between(1, 100000).fillna(False)
    dates = pd.to_datetime(values.mask(serial), format="mixed", errors="coerce")
    dates.loc[serial] = pd.to_datetime(numbers.loc[serial], unit="D", origin="1899-12-30")
    return dates


def publication_year(value):
    """提取年份，不补造月份；多个年份不能确定版本时留空。"""
    if pd.isna(value):
        return pd.NA, "missing"
    value = unicodedata.normalize("NFKC", str(value))
    if re.fullmatch(r"\d{6}|\d{8}", value):
        try:
            return pd.to_datetime(value, format="%Y%m" if len(value) == 6 else "%Y%m%d").year, "parsed"
        except ValueError:
            return pd.NA, "invalid"
    years = set(re.findall(r"(?<!\d)((?:1|2)\d{3})(?!\d)", value))
    return (int(next(iter(years))), "parsed") if len(years) == 1 else (pd.NA, "ambiguous" if years else "invalid")


def effective_missing(series):
    return series.isna() | series.astype("string").eq(UNKNOWN).fillna(False)


def profile(data, columns):
    users, books = data["读者条码"].nunique(), data["条形码"].nunique()
    events = data.dropna(subset=EVENT_KEY)
    pairs = data.dropna(subset=EVENT_KEY[:2]).drop_duplicates(EVENT_KEY[:2])
    cells = len(data) * len(columns)
    missing = sum(int(effective_missing(data[c]).sum()) for c in columns)
    times = data["处理时间"]
    divide = lambda a, b: a / b if b else None
    return {
        "rows": len(data), "users": int(users), "books": int(books),
        "titles": data.loc[~effective_missing(data["题名"]), "题名"].nunique() if "题名" in data else None,
        "unique_user_book_pairs": len(pairs),
        "repeated_user_book_events": len(data.dropna(subset=EVENT_KEY[:2])) - len(pairs),
        "duplicate_event_extras": int(events.duplicated(EVENT_KEY).sum()),
        "core_complete_rate": divide(int(data[CORE].notna().all(axis=1).sum()), len(data)),
        "core_valid_rate": divide(int((data[CORE].notna().all(axis=1) & data["操作类型"].eq("借书").fillna(False)).sum()), len(data)),
        "effective_completeness": divide(cells - missing, cells), "missing_or_unknown_cells": missing,
        "completeness_columns": len(columns),
        "interaction_density": divide(len(pairs), users * books),
        "interaction_sparsity": 1 - len(pairs) / (users * books) if users and books else None,
        "earliest_event": "" if times.dropna().empty else times.min().isoformat(sep=" "),
        "latest_event": "" if times.dropna().empty else times.max().isoformat(sep=" "),
    }


def clean_frame(raw):
    if not set(CORE).issubset(raw.columns):
        raise ValueError(f"缺少必需字段：{sorted(set(CORE) - set(raw.columns))}")
    raw = raw.copy().reset_index(drop=True)
    data = raw.copy()
    source_columns = [c for c in raw if c not in SOURCE]
    for c in source_columns:
        data[c] = raw[c].map(lambda value: normalize(value, c)).astype("string")
    original = data.copy()
    dropped = [c for c in ["流通库室键码", "附件", "架位号"] if c in data and data[c].isna().all()]
    columns = [c for c in source_columns if c not in dropped]
    checks, flags = {}, pd.Series("", index=data.index, dtype="string")

    def check(name, mask):
        mask = mask.fillna(False).astype(bool)
        checks[name] = int(mask.sum())
        flags.loc[mask] += name + ";"

    # 1. 字符、日期与关键字段检查。
    for c in source_columns:
        check(f"control_char_{c}", raw[c].map(text).str.contains(CONTROL))
    for c in ["处理时间", "外借时间", "应归还时间"]:
        if c in data:
            data[c] = parse_dates(original[c])
            check(f"invalid_datetime_{c}", original[c].notna() & data[c].isna())
    for c in EVENT_KEY:
        check(f"missing_core_{c}", original[c].isna())
    invalid = data[EVENT_KEY].isna().any(axis=1) | ~data["操作类型"].eq("借书").fillna(False)
    check("non_borrow_operation", ~data["操作类型"].eq("借书").fillna(False))
    check("outside_expected_period", data["处理时间"].notna() & ~data["处理时间"].between("2022-01-01", "2025-01-31 23:59:59.999999999"))
    for c in ["读者条码", "条形码", "登录号"]:
        if c in raw:
            check(f"numeric_identifier_precision_{c}", raw[c].map(lambda v: isinstance(v, (int, float, np.number)) and not isinstance(v, bool) and len(text(v).split(".")[0]) > 15))

    # 2. 数值异常置空，真实的 0 保留。
    for c in ["赔罚款", "超期天数"]:
        if c in data:
            number = pd.to_numeric(data[c], errors="coerce").astype("Float64")
            bad = data[c].notna() & (number.isna() | number.lt(0) | ~np.isfinite(number))
            if c == "超期天数":
                bad |= number.mod(1).ne(0).fillna(False)
            check(f"invalid_numeric_{c}", bad)
            data[c] = number.mask(bad).astype("Int64" if c == "超期天数" else "Float64")

    # 3. 年级与系别只允许使用本行明确的信息恢复。
    if "年级" in data:
        grade = pd.to_numeric(data["年级"].str.extract(r"^((?:19|20)\d{2})(?:\s*级)?$", expand=False), errors="coerce").astype("Int64")
        check("invalid_grade", original["年级"].notna() & grade.isna())
        future = grade.gt(data["处理时间"].dt.year).fillna(False)
        check("grade_after_event_year", future)
        data["年级"] = grade.mask(future)
        if "班级" in data:
            inferred = pd.to_numeric(data["班级"].str.extract(r"(?<!\d)((?:19|20)\d{2})\s*级", expand=False), errors="coerce").astype("Int64")
            recover = data["年级"].isna() & inferred.notna() & inferred.le(data["处理时间"].dt.year).fillna(False)
            data.loc[recover, "年级"] = inferred[recover]
            checks["grade_from_class"] = int(recover.sum())
    if "系别" in data and "单位" in data:
        recover = data["系别"].isna() & data["单位"].str.fullmatch(r"[^\\/]+学院(?:[（(][^）)]*[）)])?").fillna(False)
        data.loc[recover, "系别"] = data.loc[recover, "单位"]
        checks["department_from_unit"] = int(recover.sum())

    # 4. 出版年和类目是派生特征，原始出版日期、索书号保留。
    if "出版日期" in data:
        parsed = data["出版日期"].map(publication_year)
        years = pd.Series([v[0] for v in parsed], index=data.index, dtype="Int64")
        status = pd.Series([v[1] for v in parsed], index=data.index)
        check("invalid_publication_year", status.eq("invalid"))
        check("ambiguous_publication_year", status.eq("ambiguous"))
        outside = ~years.between(1450, 2025).fillna(True)
        future = years.gt(data["处理时间"].dt.year).fillna(False)
        check("publication_year_out_of_range", outside)
        check("publication_after_event_year", future)
        data["出版年份"] = years.mask(outside | future)
    if "索书号" in data:
        letter = data["索书号"].str.extract(r"^([A-Za-z])", expand=False).str.upper()
        valid = letter.isin(set("ABCDEFGHIJKNOPQRSTUVXZ"))
        data["中图法一级类目"] = letter.where(valid, UNKNOWN)
        check("unrecognized_clc", data["索书号"].notna() & ~valid)
    for c in TEXT_COLUMNS:
        if c in data:
            data[c] = data[c].fillna(UNKNOWN)

    # 5. 时间关系异常仅标记，处理时间始终作为借阅事件时间。
    if "外借时间" in data:
        check("loan_time_differs_from_event", data["外借时间"].notna() & data["处理时间"].notna() & data["外借时间"].ne(data["处理时间"]))
        check("loan_time_after_event", data["外借时间"].gt(data["处理时间"]))
    if "应归还时间" in data:
        check("due_before_event", data["应归还时间"].lt(data["处理时间"]))
        if "外借时间" in data:
            check("due_before_loan", data["应归还时间"].lt(data["外借时间"]))

    # 6. 仅删除完全业务重复；不同时间的再次借阅保留。
    duplicate = pd.Series(False, index=data.index)
    duplicate.loc[~invalid] = original.loc[~invalid].duplicated([c for c in columns if c != "序号"])
    keep = ~invalid & ~duplicate
    check("event_key_conflict", keep & data.loc[keep].duplicated(EVENT_KEY, keep=False).reindex(data.index, fill_value=False))
    for key, attributes in [("条形码", ["登录号", "题名", "责任者", "出版者", "出版日期", "索书号"]),
                            ("读者条码", ["单位", "读者级别", "班级", "年级", "专业", "系别"]), ("登录号", ["条形码"])]:
        if key in data:
            for field in attributes:
                if field in data:
                    counts = data.loc[keep & data[key].notna() & ~effective_missing(data[field])].groupby(key)[field].nunique()
                    check(f"entity_conflict_{key}_{field}", keep & data[key].isin(counts[counts.gt(1)].index))
    data["quality_flags"] = flags.str.rstrip(";")
    if "source_excel_row" not in data:
        data["source_excel_row"] = np.arange(len(data)) + 2
    cleaned = data.loc[keep].drop(columns=dropped).sort_values(["处理时间", "读者条码", "条形码", "source_excel_row"], kind="stable")
    removed = data.loc[~keep].copy()
    if len(raw) != len(cleaned) + len(removed) or cleaned[CORE].isna().any().any():
        raise RuntimeError("清洗后的记录数量或关键字段校验失败。")

    # 7. 质量指标使用同一组原始字段计算；“未知”仍计为缺失。
    field_quality = []
    for c in source_columns:
        output = cleaned[c] if c in cleaned else pd.Series(dtype="string")
        field_quality.append({"column": c, "dropped": c in dropped,
                              "raw_effective_missing": int(original[c].isna().sum()),
                              "raw_missing_rate": original[c].isna().mean() if len(raw) else None,
                              "clean_null_cells": int(output.isna().sum()) if c not in dropped else None,
                              "clean_unknown_markers": int(output.astype("string").eq(UNKNOWN).fillna(False).sum()) if c not in dropped else None,
                              "clean_effective_missing": int(effective_missing(output).sum()) if c not in dropped else None,
                              "clean_missing_rate": effective_missing(output).mean() if len(cleaned) and c not in dropped else None})
    summary = {"before": profile(original.assign(处理时间=data["处理时间"]), columns), "after": profile(cleaned, columns), "checks": checks,
               "rejected_rows": int(invalid.sum()), "duplicate_rows_removed": int(duplicate.sum()),
               "retention_rate": len(cleaned) / len(raw) if len(raw) else None,
               "row_reconciliation_passed": len(raw) == len(cleaned) + len(removed)}
    return {"cleaned": cleaned, "removed": removed, "summary": summary, "column_quality": pd.DataFrame(field_quality)}


def save_csv(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = data.copy()
    for c in data:
        if pd.api.types.is_datetime64_any_dtype(data[c]):
            data[c] = data[c].map(lambda v: "" if pd.isna(v) else v.isoformat(sep=" "))
    data.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def run(input_path, output_dir, sheet=None):
    result = clean_frame(read_source(input_path, sheet))
    output_dir = Path(output_dir)
    save_csv(result["cleaned"], output_dir / "borrowing_cleaned.csv")
    save_csv(result["column_quality"], output_dir / "field_quality.csv")
    metrics = []
    for group, values in result["summary"].items():
        if isinstance(values, dict):
            metrics.extend({"category": group, "metric": key, "value": value} for key, value in values.items())
        else:
            metrics.append({"category": "processing", "metric": group, "value": values})
    save_csv(pd.DataFrame(metrics), output_dir / "quality_metrics.csv")
    print(f"预处理完成：{result['summary']['before']['rows']} → {len(result['cleaned'])} 条记录")
    print(f"输出目录：{output_dir.resolve()}")
    return result


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=base / "data" / "2022-2025处理后借阅信息.xlsx")
    parser.add_argument("--output-dir", type=Path, default=base / "data" / "processed")
    parser.add_argument("--sheet", default=None)
    args = parser.parse_args()
    run(args.input, args.output_dir, args.sheet)
