#!/usr/bin/env python3
"""Generate Journal-of-Big-Data-style cohort baseline table (LaTeX + CSV)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
DATA = REPO / "data"
COHORT_ROOT = DATA / "All_3000_5cens"
REGISTRY = DATA / "colposcopy_3000" / "3000_nums.xlsx"
COL_RAW = DATA / "colposcopy_3000"
OUT_DIR = REPO / "experiments/exp_infofusion_2026/paper_revision/tables"

HR_HPV_GENOTYPES = {
    "16", "18", "31", "33", "35", "39", "45", "51", "52", "53", "56", "58", "59", "66", "68", "73", "82", "81",
}

CENTER_EN = {
    "武大人民医院": ("Renmin Hospital of Wuhan University", 1),
    "恩施州中心医院": ("Enshi Central Hospital", 2),
    "襄阳市中心医院": ("Xiangyang Central Hospital", 3),
    "十堰市人民医院": ("Shiyan People's Hospital", 4),
    "荆州市第一人民医院": ("Jingzhou First People's Hospital", 5),
}

RAW_CENTER_DIRS = {
    center: COL_RAW / f"阴道镜图像（{center}）"
    for center in CENTER_EN
}

REPORT_AVAILABILITY = {
    "武大人民医院": ("No", "Colposcopy still images only; no structured examination report PDF/XML in archive"),
    "十堰市人民医院": ("No", "Colposcopy still images only; no examination report files in archive"),
    "恩施州中心医院": ("Yes", "PDF colposcopy reports and ExamResult.xml in Enshi patient folders"),
    "襄阳市中心医院": ("Sparse", "Few PDF/XML reports; majority folders contain still images only"),
    "荆州市第一人民医院": ("Yes", "report.jpg and Report.ini per examination folder (~623 cases in archive)"),
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def load_cohort() -> pd.DataFrame:
    tr = pd.read_csv(COHORT_ROOT / "train_labels.csv")
    te = pd.read_csv(COHORT_ROOT / "test_labels.csv")
    tr["split"] = "train"
    te["split"] = "test"
    return pd.concat([tr, te], ignore_index=True)


def classify_hpv(val) -> str:
    if pd.isna(val) or str(val).strip() in ("", "nan", "-", "—"):
        return "unclassifiable"
    s = str(val).strip()
    if s in {"阴性", "negative", "Negative", "NEGATIVE"}:
        return "negative"
    if "高危" in s and not re.search(r"\d", s):
        return "hr_positive"
    nums = re.findall(r"\d+", s)
    if nums and any(n in HR_HPV_GENOTYPES for n in nums):
        return "hr_positive"
    if nums:
        return "unclassifiable"
    return "unclassifiable"


def classify_tct(val) -> str:
    if pd.isna(val) or str(val).strip() in ("", "nan", "-", "—"):
        return "missing"
    s = str(val).strip().upper()
    if s in {"NILM", "NORMAL", "WNL"} or "NILM" in s:
        return "negative"
    if "ASC-US" in s:
        return "asc_us"
    if any(k in s for k in ("LSIL", "ASC-H", "HSIL", "AGC", "SCC", "癌", "MALIGNANT", "CA")):
        return "lsil_or_worse"
    return "missing"


def _pathology_text(*fields) -> str:
    return " ".join(
        str(x).strip()
        for x in fields
        if pd.notna(x) and str(x).strip() not in ("", "nan", "None")
    )


def _strip_negated_cancer_phrases(text: str) -> str:
    cleaned = text
    for pat in (
        r"未见[^。；;，,]{0,40}癌[^。；;，,]*",
        r"无[^。；;，,]{0,30}癌[^。；;，,]*",
        r"阴性[^。；;，,]{0,20}癌[^。；;，,]*",
    ):
        cleaned = re.sub(pat, " ", cleaned)
    return cleaned


def _has_biopsy_invasive_cancer(text: str) -> bool:
    text = _strip_negated_cancer_phrases(text)
    return bool(
        re.search(
            r"浸润|微小浸润|鳞状细胞癌|鳞癌|腺癌|低分化癌|中[-—]?低分化.*癌|恶性肿瘤|宫颈癌|癌栓|癌转移",
            text,
        )
    )


def _has_followup_cervical_invasive_cancer(text: str) -> bool:
    text = _strip_negated_cancer_phrases(text)
    if re.search(
        r"(宫颈|子宫颈|颈管|宫颈锥切|LEEP)[^。；;]{0,120}"
        r"(浸润|微小浸润|鳞状细胞癌|鳞癌|腺癌|低分化癌|恶性肿瘤|癌栓|癌转移|宫颈癌)",
        text,
    ):
        return True
    return bool(re.search(r"浸润性鳞状细胞癌|宫颈癌", text))


def _has_explicit_cin3(text: str) -> bool:
    return bool(
        re.search(
            r"CIN\s*(?:3|III|Ⅲ|三)"
            r"|CIN\s*(?:2|II|Ⅱ|二)\s*[-~～—至]+\s*(?:3|III|Ⅲ|三)"
            r"|上皮内癌|原位癌",
            text,
            re.I,
        )
    )


def _has_explicit_cin2(text: str) -> bool:
    return bool(
        re.search(
            r"CIN\s*(?:2|II|Ⅱ|二)(?!\s*[-~～—至]+\s*(?:3|III|Ⅲ|三))",
            text,
            re.I,
        )
    )


def _has_ungraded_hsil(text: str) -> bool:
    high_grade = any(
        re.search(pat, text, re.I)
        for pat in (
            r"CIN\s*高[级級]别",
            r"高[级級]别\s*CIN",
            r"宫颈高[级級]别病变",
            r"(?:宫颈|子宫颈|颈管|宫颈物|宫颈组织)[^。；;]{0,80}"
            r"高[级級]别(?:鳞状)?上皮内(?:病变|瘤变)",
            r"HSIL|高度鳞状上皮内(?:病变|瘤变)",
        )
    )
    if not high_grade:
        return False

    explicit_high_grade_cin = bool(
        re.search(r"CIN\s*高[级級]别|高[级級]别\s*CIN|宫颈高[级級]别病变", text, re.I)
    )
    suspicious = bool(re.search(r"可疑|不能排除|建议.{0,20}免疫组化|除外", text))
    non_cervical = bool(re.search(r"(阴道|VAIN|VA1N|外阴|VIN)[^。；;]{0,60}高[级級]别", text, re.I))
    cervical_high_grade = bool(
        re.search(
            r"(?:宫颈|子宫颈|颈管|宫颈物|宫颈组织)[^。；;]{0,80}高[级級]别",
            text,
        )
    )
    if suspicious and not explicit_high_grade_cin:
        return False
    if non_cervical and not cervical_high_grade:
        return False
    return True


def classify_histology(path_result, path_grade=None, oct_label=None, treatment=None) -> str:
    primary_text = _pathology_text(path_result, path_grade)
    followup_text = _pathology_text(treatment)
    text = _pathology_text(path_result, path_grade, treatment)
    if not text:
        return "missing"

    if _has_biopsy_invasive_cancer(primary_text) or _has_followup_cervical_invasive_cancer(followup_text):
        return "invasive"
    if _has_explicit_cin3(text):
        return "cin3"
    if _has_explicit_cin2(text):
        return "cin2"
    if _has_ungraded_hsil(text):
        # Conservative split-table bucket: ungraded HSIL/high-grade CIN is CIN2+,
        # but not promoted to CIN3+ without explicit CIN3/AIS wording.
        return "cin2"
    if re.search(r"CIN\s*(?:1|I|Ⅰ|一)|低级别|低度.*(?:瘤变|病变)|LSIL", text, re.I):
        return "cin0_1"
    if re.search(r"湿疣|慢性[子]?宫颈炎|鳞状上皮化生|炎症", text, re.I):
        return "cin0_1"
    if oct_label == 1:
        return "cin0_1"
    if oct_label == 0:
        return "cin0_1"
    return "missing"


def col_folder_has_report(col_path: Path) -> bool:
    if not col_path.exists():
        return False
    for p in col_path.rglob("*"):
        if not p.is_file():
            continue
        n = p.name.lower()
        if p.suffix.lower() in {".pdf", ".xml", ".doc", ".docx"}:
            return True
        if n in {"report.jpg", "report.ini"} or "检查报告" in p.name:
            return True
    return False


def is_counted_image_file(path: Path, modality: str) -> bool:
    if path.suffix.lower() not in IMAGE_EXTS:
        return False
    if modality == "colposcopy" and path.name.lower() in {"report.jpg", "report.jpeg", "report.png"}:
        return False
    return True


def count_image_files(path: Path | None, modality: str) -> int:
    if path is None or not path.exists():
        return 0
    return sum(
        1
        for item in path.rglob("*")
        if item.is_file() and is_counted_image_file(item, modality)
    )


def _normalise_patient_name(name: str) -> str:
    return re.sub(r"^[\W_·•.。\s]+|[\W_·•.。\s]+$", "", str(name).strip())


def _cohort_date_name_key(case_id: str):
    parts = str(case_id).split("_", 1)
    if len(parts) != 2:
        return None
    date = re.sub(r"\D", "", parts[0])
    if len(date) < 8:
        return None
    return date[:8], _normalise_patient_name(parts[1])


def build_enshi_archive_index() -> dict[tuple[str, str], list[Path]]:
    """Map All_3000_5cens Enshi IDs (YYYYMMDD_name) to raw folders.

    The released colposcopy symlinks point at an image-only legacy copy, while
    the source archive keeps reports under folders such as YYYYMMDD001_name.
    """
    root = RAW_CENTER_DIRS["恩施州中心医院"]
    index: dict[tuple[str, str], list[Path]] = {}
    if not root.exists():
        return index
    for case_dir in root.iterdir():
        if not case_dir.is_dir():
            continue
        m = re.match(r"^(\d{8})\d*_(.+)$", case_dir.name)
        if not m:
            continue
        key = (m.group(1), _normalise_patient_name(m.group(2)))
        index.setdefault(key, []).append(case_dir)
    return index


def raw_archive_dirs_for_row(row: pd.Series, enshi_archive_index: dict[tuple[str, str], list[Path]]) -> list[Path]:
    if row.get("center_name") != "恩施州中心医院":
        return []
    key = _cohort_date_name_key(row.get("ID", ""))
    if key is None:
        return []
    return enshi_archive_index.get(key, [])


def pct(n: int, denom: int) -> str:
    if denom == 0:
        return "0 (0.0)"
    return f"{n} ({n / denom * 100:.1f})"


def mean_sd(series: pd.Series) -> str:
    s = pd.to_numeric(series, errors="coerce")
    s = s[(s >= 15) & (s <= 90)]
    s = s.dropna()
    if len(s) == 0:
        return "—"
    return f"{s.mean():.1f} ({s.std(ddof=1):.1f})"


def summarize_group(df: pd.DataFrame) -> dict:
    n = len(df)
    hpv = df["hpv_class"].value_counts()
    tct = df["tct_class"].value_counts()
    hist = df["hist_class"].value_counts()
    rep = int(df["has_colposcopy_report"].sum())
    oct_images = int(df["oct_image_count"].sum())
    col_images = int(df["colposcopy_image_count"].sum())
    return {
        "n": n,
        "oct_image_n": oct_images,
        "colposcopy_image_n": col_images,
        "total_image_n": oct_images + col_images,
        "age": mean_sd(df["AGE"]),
        "hpv_pos": int(hpv.get("hr_positive", 0)),
        "hpv_neg": int(hpv.get("negative", 0)),
        "hpv_unc": int(hpv.get("unclassifiable", 0)),
        "tct_neg": int(tct.get("negative", 0)),
        "tct_ascus": int(tct.get("asc_us", 0)),
        "tct_lsil": int(tct.get("lsil_or_worse", 0)),
        "tct_miss": int(tct.get("missing", 0)),
        "cin01": int(hist.get("cin0_1", 0)),
        "cin2": int(hist.get("cin2", 0)),
        "cin3": int(hist.get("cin3", 0)),
        "inv": int(hist.get("invasive", 0)),
        "hist_miss": int(hist.get("missing", 0)),
        "oct_pos": int((df["label"] == 1).sum()),
        "oct_neg": int((df["label"] == 0).sum()),
        "report_n": rep,
        "report_pct": rep / n * 100 if n else 0,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cohort = load_cohort()
    mi = pd.read_excel(REGISTRY, sheet_name="MedicalInfo").drop_duplicates("OCT图像Id")
    mi = mi.rename(
        columns={
            "OCT图像Id": "OCT",
            "HPV清洗（高亮表示阳性）": "HPV_reg",
            "TCT清洗（高亮表示阳性）": "TCT_reg",
            "年龄": "AGE_reg",
        }
    )

    df = cohort.merge(
        mi[["OCT", "HPV_reg", "TCT_reg", "AGE_reg", "病理结果", "病理级别", "后续治疗", "OCT二次判读", "医院"]],
        on="OCT",
        how="left",
        suffixes=("", "_dup"),
    )
    df["AGE"] = pd.to_numeric(df["AGE"].fillna(df["AGE_reg"]), errors="coerce")
    df["HPV_use"] = df["HPV清洗"].where(df["HPV清洗"].notna() & (df["HPV清洗"].astype(str).str.strip() != ""), df["HPV_reg"])
    df["TCT_use"] = df["TCT清洗"].where(df["TCT清洗"].notna() & (df["TCT清洗"].astype(str).str.strip() != ""), df["TCT_reg"])

    df["hpv_class"] = df["HPV_use"].map(classify_hpv)
    df["tct_class"] = df["TCT_use"].map(classify_tct)
    df["hist_class"] = [
        classify_histology(pr, pg, lb, tx)
        for pr, pg, lb, tx in zip(df["病理结果"], df["病理级别"], df["label"], df["后续治疗"])
    ]

    # Report availability is a source-archive property. Enshi's release symlinks
    # point to image-only folders, so recover the raw patient folder first.
    enshi_archive_index = build_enshi_archive_index()
    report_flags = []
    report_sources = []
    report_checked_paths = []
    for _, row in df.iterrows():
        raw_report_dirs = raw_archive_dirs_for_row(row, enshi_archive_index)
        raw_report_dir = next((p for p in raw_report_dirs if col_folder_has_report(p)), None)
        if raw_report_dir is not None:
            report_flags.append(True)
            report_sources.append("source_archive")
            report_checked_paths.append(str(raw_report_dir))
            continue

        col_dir = None
        for split in ("train", "test"):
            p = COHORT_ROOT / split / "col" / row["ID"]
            if p.exists():
                col_dir = p.resolve()
                break
        has_report = col_folder_has_report(col_dir) if col_dir else False
        report_flags.append(has_report)
        report_sources.append("release_col_link" if col_dir else "not_found")
        report_checked_paths.append(str(col_dir) if col_dir else "")
    df["has_colposcopy_report"] = report_flags
    df["colposcopy_report_source"] = report_sources
    df["colposcopy_report_checked_path"] = report_checked_paths

    image_count_cache: dict[tuple[str, str, str], int] = {}
    oct_image_counts = []
    colposcopy_image_counts = []
    for _, row in df.iterrows():
        split = str(row["split"])
        oct_id = str(row["OCT"])
        case_id = str(row["ID"])
        oct_key = ("oct", split, oct_id)
        col_key = ("colposcopy", split, case_id)
        if oct_key not in image_count_cache:
            image_count_cache[oct_key] = count_image_files(COHORT_ROOT / split / "oct" / oct_id, "oct")
        if col_key not in image_count_cache:
            image_count_cache[col_key] = count_image_files(COHORT_ROOT / split / "col" / case_id, "colposcopy")
        oct_image_counts.append(image_count_cache[oct_key])
        colposcopy_image_counts.append(image_count_cache[col_key])
    df["oct_image_count"] = oct_image_counts
    df["colposcopy_image_count"] = colposcopy_image_counts
    df["total_image_count"] = df["oct_image_count"] + df["colposcopy_image_count"]

    rows = []
    for center_zh, (center_en, order) in sorted(CENTER_EN.items(), key=lambda x: x[1]):
        sub = df[df["center_name"] == center_zh]
        s = summarize_group(sub)
        rep_flag, rep_note = REPORT_AVAILABILITY[center_zh]
        rows.append(
            {
                "order": order,
                "centre_en": center_en,
                "centre_zh": center_zh,
                "archive_report": rep_flag,
                "archive_report_note": rep_note,
                "in_cohort_with_report_file": s["report_n"],
                "in_cohort_report_pct": round(s["report_pct"], 1),
                **s,
            }
        )

    overall = summarize_group(df)
    rows.append(
        {
            "order": 99,
            "centre_en": "Overall",
            "centre_zh": "合计",
            "archive_report": "Partial",
            "archive_report_note": "See centre-specific notes",
            "in_cohort_with_report_file": overall["report_n"],
            "in_cohort_report_pct": round(overall["report_pct"], 1),
            **overall,
        }
    )

    tab = pd.DataFrame(rows).sort_values("order")
    tab.to_csv(OUT_DIR / "cohort_baseline_by_center.csv", index=False, encoding="utf-8-sig")
    df.to_csv(OUT_DIR / "cohort_baseline_patient_level.csv", index=False, encoding="utf-8-sig")

    # Registry screened counts
    reg = mi.copy()
    reg_stats = reg.groupby("医院").size().reset_index(name="registry_screened_n")

    latex_lines = []
    latex_lines.append(r"% Auto-generated by generate_cohort_baseline_table.py")
    latex_lines.append(r"\begin{landscape}")
    latex_lines.append(r"\begin{center}")
    latex_lines.append(r"  \begin{minipage}{1\linewidth}")
    latex_lines.append(r"    \captionsetup{type=table}")
    latex_lines.append(
        r"    \captionof{table}{Baseline characteristics of the final five-centre multimodal analytic cohort, overall and by participating centre}"
    )
    latex_lines.append(r"    \label{tab:baseline_all3000_5cens}")
    latex_lines.append(r"    \normalsize")
    latex_lines.append(r"    \setlength{\tabcolsep}{2pt}")
    latex_lines.append(r"    \renewcommand{\arraystretch}{1.8}")
    latex_lines.append(r"    \begin{adjustbox}{width=\linewidth}")
    latex_lines.append(
        r"    \begin{tabular}{c >{\raggedright\arraybackslash}p{4.2cm} c c c c "
        r"*{12}{c}}"
    )
    latex_lines.append(r"      \toprule")
    latex_lines.append(
        r"      \textbf{No.} & \textbf{Participating centre} & "
        r"\makecell[c]{\textbf{Archived}\\ \textbf{colposcopy}\\ \textbf{report}} & "
        r"\makecell[c]{\textbf{Cases with}\\ \textbf{archived report,}\\ \textbf{n (\%)}} & "
        r"\makecell[c]{\textbf{Examinations,}\\ \textbf{n}} & "
        r"\makecell[c]{\textbf{Total images,}\\ \textbf{n}} & "
        r"\makecell[c]{\textbf{Age, years,}\\ \textbf{mean (SD)}} & "
        r"\makecell[c]{\textbf{hrHPV positive}\\ \textbf{by prespecified}\\ \textbf{genotype panel,}\\ \textbf{n (\%)}} & "
        r"\makecell[c]{\textbf{HPV status}\\ \textbf{unavailable or}\\ \textbf{unclassifiable,}\\ \textbf{n (\%)}} & "
        r"\makecell[c]{\textbf{Negative}\\ \textbf{cytology,}\\ \textbf{n (\%)}} & "
        r"\makecell[c]{\textbf{ASC-US,}\\ \textbf{n (\%)}} & "
        r"\makecell[c]{\textbf{LSIL or}\\ \textbf{worse,}\\ \textbf{n (\%)}} & "
        r"\makecell[c]{\textbf{Missing}\\ \textbf{cytology,}\\ \textbf{n (\%)}} & "
        r"\makecell[c]{\textbf{Positive}\\ \textbf{OCT--clinical}\\ \textbf{training label,}\\ \textbf{n (\%)}} & "
        r"\makecell[c]{\textbf{CIN0--1 or}\\ \textbf{mapped benign}\\ \textbf{histology,}\\ \textbf{n (\%)}} & "
        r"\makecell[c]{\textbf{CIN2 or}\\ \textbf{ungraded}\\ \textbf{HSIL,}\\ \textbf{n (\%)}} & "
        r"\makecell[c]{\textbf{CIN3+,}\\ \textbf{n (\%)}} & "
        r"\makecell[c]{\textbf{Invasive}\\ \textbf{cervical cancer,}\\ \textbf{n (\%)}} \\"
    )
    latex_lines.append(r"      \midrule")

    for _, r in tab.iterrows():
        n = int(r["n"])
        if r["centre_en"] == "Overall":
            latex_lines.append(r"      \rowcolor{gray!10}")
        cin3p = int(r["cin3"]) + int(r["inv"])
        line = (
            f"      {int(r['order']) if r['order']<90 else ''} & {r['centre_en']} & "
            f"{r['archive_report']} & {pct(int(r['in_cohort_with_report_file']), n)} & "
            f"{n} & {int(r['total_image_n']):,} & {r['age']} & "
            f"{pct(int(r['hpv_pos']), n)} & {pct(int(r['hpv_unc']), n)} & "
            f"{pct(int(r['tct_neg']), n)} & {pct(int(r['tct_ascus']), n)} & {pct(int(r['tct_lsil']), n)} & "
            f"{pct(int(r['tct_miss']), n)} & "
            f"{pct(int(r['oct_pos']), n)} & "
            f"{pct(int(r['cin01']), n)} & {pct(int(r['cin2']), n)} & "
            f"{pct(int(r['cin3']), n)} & {pct(int(r['inv']), n)} \\\\"
        )
        latex_lines.append(line)

    latex_lines.append(r"      \bottomrule")
    latex_lines.append(r"    \end{tabular}")
    latex_lines.append(r"    \end{adjustbox}")
    latex_lines.append(r"    \vspace{0.4em}")
    latex_lines.append(r"    \begin{minipage}{\linewidth}")
    latex_lines.append(r"      \footnotesize")
    latex_lines.append(
        r"      \textit{The final analytic cohort comprised examinations with paired OCT volumes, colposcopy still images, and clinical priors linked to administrative registry records. "
        r"The registry included 3,010 screened examinations and 3,009 unique OCT identifiers after de-duplication, from which a final multimodal cohort of 1,897 examinations with verified imaging was derived. "
        r"Images denote the total number of OCT B-scan image files plus colposcopy still-image files available for the final analytic cohort; archived report images, Report.ini files, and pixel-guidance masks were not counted. "
        r"Archived colposcopy-report availability describes the source archives only and was not used as a model input. "
        r"Centres classified as ``Yes'' had structured report exports available (e.g., report images from Jingzhou and PDF/XML records from Enshi). "
        r"Wuhan and Shiyan archives contained colposcopy images only, whereas Xiangyang had sparse report availability. "
        r"``Cases with archived report'' denotes examinations for which at least one structured report file was present in the clinical archive. "
        r"hrHPV positivity was defined according to the prespecified high-risk genotype panel: HPV 16, 18, 31, 33, 35, 39, 45, 51, 52, 53, 56, 58, 59, 66, 68, 73, 81, or 82. "
        r"The positive OCT--clinical training label denotes the harmonised binary supervision label used for multimodal model training and was derived from OCT second-read text when definitive pathology mapping was unavailable. "
        r"CIN categories were harmonised from clinical pathology reports and downstream treatment or excision records when available. "
        r"Ungraded high-grade CIN or HSIL without explicit CIN3 or AIS terminology was conservatively assigned to the CIN2/HSIL category; explicit CIN2--3 was classified as CIN3+. "
        r"Invasive cervical cancer required explicit invasive-cancer terminology in a cervical context or corroborating downstream excision records.}"
    )
    latex_lines.append(r"    \end{minipage}")
    latex_lines.append(r"  \end{minipage}")
    latex_lines.append(r"\end{center}")
    latex_lines.append(r"\end{landscape}")

    (OUT_DIR / "cohort_baseline_table.tex").write_text("\n".join(latex_lines), encoding="utf-8")

    # Markdown narrative for user
    md = []
    md.append("# Five-centre cohort description (Journal of Big Data)\n")
    md.append(f"## Final multimodal analytic cohort (`All_3000_5cens`)\n")
    md.append(f"- **Total examinations with paired OCT + colposcopy**: **{len(df)}**")
    md.append(f"- **Total image files counted in the analytic cohort**: **{overall['total_image_n']:,}** (OCT B-scans {overall['oct_image_n']:,}; colposcopy stills {overall['colposcopy_image_n']:,})")
    md.append(f"- **Train / test**: {len(cohort[cohort.split=='train'])} / {len(cohort[cohort.split=='test'])}")
    md.append(f"- **Registry screened (unique OCT ID)**: 3009")
    md.append(f"- **Binary label (OCT-based)**: negative {overall['oct_neg']} ({overall['oct_neg']/len(df)*100:.1f}%), positive {overall['oct_pos']} ({overall['oct_pos']/len(df)*100:.1f}%)\n")
    md.append("## Colposcopy examination report availability in source archives\n")
    for czh, (cen, _) in CENTER_EN.items():
        flag, note = REPORT_AVAILABILITY[czh]
        sub_n = len(df[df.center_name == czh])
        rep_n = int(df[df.center_name == czh]["has_colposcopy_report"].sum())
        md.append(f"- **{cen}** ({czh}): archive **{flag}** — {note}. In analytic cohort: **{rep_n}/{sub_n}** ({rep_n/sub_n*100:.1f}%) source/linked folders contain a report file.\n")
    md.append("\n## Per-centre table\n")
    md.append(tab.to_csv(index=False))
    (OUT_DIR / "cohort_baseline_narrative.md").write_text("\n".join(md), encoding="utf-8")

    print(json.dumps({"cohort_n": len(df), "output_dir": str(OUT_DIR)}, indent=2))
    print(tab.to_string())


if __name__ == "__main__":
    main()
