#!/usr/bin/env python3
"""Supplementary Table S1 (registry cohort) + landscape CONSORT TikZ."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# Reuse harmonisation from main table generator
from generate_cohort_baseline_table import (  # type: ignore
    CENTER_EN,
    COHORT_ROOT,
    DATA,
    REGISTRY,
    REPORT_AVAILABILITY,
    classify_histology,
    classify_hpv,
    classify_tct,
    pct,
    summarize_group,
)


def infer_label(oct_reading) -> int | None:
    if pd.isna(oct_reading) or str(oct_reading).strip() in ("", "nan"):
        return None
    s = str(oct_reading)
    if "高级别" in s or "疑似" in s:
        return 1
    if "未发现" in s or "低级别" in s:
        return 0
    return None

REPO = Path(__file__).resolve().parents[4]
OUT_DIR = REPO / "experiments/exp_infofusion_2026/paper_revision/tables"
FIG_DIR = REPO / "experiments/exp_infofusion_2026/paper_revision/figures"
OCT_REMOTE = DATA / "cervix_oct_original"

HOSPITAL_OCT_CENTER = {
    "荆州市第一人民医院": "jingzhou",
    "武大人民医院": "WuDa",
    "十堰市人民医院": "ShiYan",
    "襄阳市中心医院": "XiangYang",
    "恩施州中心医院": "Enshi",
}
OCT_PREFIX_CENTER = {
    "M0008": "jingzhou",
    "M22101": "ShiYan",
    "M20105": "WuDa",
    "M20203": "WuDa",
    "M22104": "ShiYan",
    "M22102": "XiangYang",
    "M22105": "Enshi",
}


def load_registry() -> pd.DataFrame:
    mi = pd.read_excel(REGISTRY, sheet_name="MedicalInfo")
    oct_img = pd.read_excel(REGISTRY, sheet_name="OCTImages")
    mi = mi.drop_duplicates(subset=["OCT图像Id"], keep="first")
    use_cols = [c for c in ("OCT图像Id", "OCT二次判读", "OCT实时判读", "二次判读疑似", "二次判读高级别") if c in oct_img.columns]
    oct_img = oct_img[use_cols].rename(
        columns={"OCT二次判读": "OCT二次判读_img", "OCT实时判读": "OCT实时判读_img"}
    )
    df = mi.merge(oct_img, on="OCT图像Id", how="left")
    df = df.drop_duplicates(subset=["OCT图像Id"], keep="first")
    df = df.rename(
        columns={
            "OCT图像Id": "OCT",
            "医院": "center_name",
            "年龄": "AGE",
            "HPV清洗（高亮表示阳性）": "HPV_use",
            "TCT清洗（高亮表示阳性）": "TCT_use",
            "OCT二次判读": "OCT_read_mi",
        }
    )
    df["AGE"] = pd.to_numeric(df["AGE"], errors="coerce")
    df["hpv_class"] = df["HPV_use"].map(classify_hpv)
    df["tct_class"] = df["TCT_use"].map(classify_tct)

    def oct_abnormal(row) -> int | None:
        for col in ("OCT二次判读_img", "OCT_read_mi", "OCT实时判读_img"):
            if col in row.index:
                lab = infer_label(row[col])
                if lab is not None:
                    return lab
        if "二次判读高级别" in row.index and pd.notna(row["二次判读高级别"]):
            try:
                if float(row["二次判读高级别"]) > 0:
                    return 1
            except (TypeError, ValueError):
                pass
        return None

    df["oct_abnormal"] = df.apply(oct_abnormal, axis=1)
    df["hist_class"] = [
        classify_histology(pr, pg, oa if oa is not None else 0, tx)
        for pr, pg, oa, tx in zip(
            df["病理结果"],
            df["病理级别"],
            df["oct_abnormal"].fillna(0),
            df["后续治疗"],
        )
    ]
    return df


def count_oct_remote(oct_id: str, hospital: str) -> bool:
    centers = []
    if hospital in HOSPITAL_OCT_CENTER:
        centers.append(HOSPITAL_OCT_CENTER[hospital])
    prefix = str(oct_id).split("_")[0]
    if prefix in OCT_PREFIX_CENTER:
        centers.append(OCT_PREFIX_CENTER[prefix])
    for c in dict.fromkeys(centers):
        p = OCT_REMOTE / c / str(oct_id)
        if p.is_dir():
            return True
        cd = OCT_REMOTE / c
        if cd.exists() and list(cd.glob(f"{oct_id}*")):
            return True
    return False


def consort_counts(reg: pd.DataFrame) -> dict[str, int]:
    n_screened = int(pd.read_excel(REGISTRY, sheet_name="MedicalInfo").shape[0])
    n_unique = len(reg)
    n_dup = n_screened - n_unique

    analytic = pd.concat(
        [
            pd.read_csv(COHORT_ROOT / "train_labels.csv"),
            pd.read_csv(COHORT_ROOT / "test_labels.csv"),
        ]
    )
    analytic_ids = set(analytic["OCT"].astype(str))
    n_multimodal = len(analytic_ids)

    oct_avail = sum(count_oct_remote(oid, hosp) for oid, hosp in zip(reg["OCT"], reg["center_name"]))
    n_no_oct = n_unique - oct_avail
    n_no_multimodal = n_unique - n_multimodal
    n_no_col_or_link = n_no_multimodal - n_no_oct  # approximate partition

    return {
        "screened": n_screened,
        "dup_excluded": n_dup,
        "unique_registry": n_unique,
        "oct_storage_available": oct_avail,
        "excluded_oct_missing": n_no_oct,
        "final_multimodal_paired": n_multimodal,
        "excluded_not_multimodal": n_no_multimodal,
        "excluded_col_or_alignment": max(0, n_no_col_or_link),
    }


def build_s1_rows(reg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    order_map = {zh: i for zh, (_, i) in CENTER_EN.items()}

    def reg_summarize(sub: pd.DataFrame) -> dict:
        base = summarize_group(
            sub.assign(
                label=sub["oct_abnormal"].fillna(0).astype(int),
                has_colposcopy_report=False,
            )
        )
        base["oct_abnormal_n"] = int(sub["oct_abnormal"].eq(1).sum())
        base["oct_normal_n"] = int(sub["oct_abnormal"].eq(0).sum())
        base["oct_read_missing"] = int(sub["oct_abnormal"].isna().sum())
        return base

    for center_zh in sorted(reg["center_name"].unique(), key=lambda x: order_map.get(x, 99)):
        sub = reg[reg["center_name"] == center_zh]
        center_en = CENTER_EN.get(center_zh, (center_zh, 0))[0]
        rep_flag = REPORT_AVAILABILITY.get(center_zh, ("Unknown", ""))[0]
        s = reg_summarize(sub)
        rows.append(
            {
                "order": order_map.get(center_zh, 50),
                "centre_en": center_en,
                "centre_zh": center_zh,
                "archive_report": rep_flag,
                **s,
            }
        )

    overall = reg_summarize(reg)
    rows.append(
        {
            "order": 99,
            "centre_en": "Overall",
            "centre_zh": "合计",
            "archive_report": "Partial",
            **overall,
        }
    )
    return pd.DataFrame(rows).sort_values("order")


def latex_s1_table(tab: pd.DataFrame) -> str:
    lines = [
        r"% Auto-generated: generate_cohort_supplementary_s1_consort.py",
        r"\begin{center}",
        r"  \begin{minipage}{1\linewidth}",
        r"    \captionsetup{type=table}",
        r"    \captionof{table}{Supplementary Table S1. Baseline characteristics of the screened administrative registry cohort by centre (\textit{without} multimodal linkage filtering)}",
        r"    \label{tab:supp_s1_registry}",
        r"    \normalsize",
        r"    \setlength{\tabcolsep}{2pt}",
        r"    \renewcommand{\arraystretch}{1.8}",
        r"    \begin{adjustbox}{width=\linewidth}",
        r"    \begin{tabular}{c >{\raggedright\arraybackslash}p{4.5cm} c c c *{12}{c}}",
        r"      \toprule",
        r"      \textbf{No.} & \textbf{Centre} & "
        r"\makecell[c]{\textbf{Colposcopy}\\ \textbf{report in}\\ \textbf{archive?}} & "
        r"\makecell[c]{\textbf{Registry}\\ \textbf{examinations,}\\ \textbf{n}} & "
        r"\makecell[c]{\textbf{Age, years,}\\ \textbf{mean (SD)}} & "
        r"\makecell[c]{\textbf{Prespecified}\\ \textbf{hrHPV}\\ \textbf{positive,}\\ \textbf{n (\%)}} & "
        r"\makecell[c]{\textbf{HPV}\\ \textbf{negative,}\\ \textbf{n (\%)}} & "
        r"\makecell[c]{\textbf{Unclassifiable}\\ \textbf{HPV, n (\%)}} & "
        r"\makecell[c]{\textbf{Negative}\\ \textbf{cytology,}\\ \textbf{n (\%)}} & "
        r"\makecell[c]{\textbf{ASC-US,}\\ \textbf{n (\%)}} & "
        r"\makecell[c]{\textbf{LSIL or}\\ \textbf{worse,}\\ \textbf{n (\%)}} & "
        r"\makecell[c]{\textbf{Missing}\\ \textbf{cytology,}\\ \textbf{n (\%)}} & "
        r"\makecell[c]{\textbf{OCT second-read}\\ \textbf{abnormal,}\\ \textbf{n (\%)}} & "
        r"\makecell[c]{\textbf{OCT second-read}\\ \textbf{missing,}\\ \textbf{n (\%)}} & "
        r"\makecell[c]{\textbf{CIN0--1 or}\\ \textbf{mapped}\\ \textbf{benign,}\\ \textbf{n (\%)}} & "
        r"\makecell[c]{\textbf{CIN2 or}\\ \textbf{ungraded}\\ \textbf{HSIL,}\\ \textbf{n (\%)}} & "
        r"\makecell[c]{\textbf{CIN3+,}\\ \textbf{n (\%)}} & "
        r"\makecell[c]{\textbf{Invasive}\\ \textbf{cancer,}\\ \textbf{n (\%)}} \\",
        r"      \midrule",
    ]
    no = 0
    for _, r in tab.iterrows():
        n = int(r["n"])
        if r["centre_en"] == "Overall":
            lines.append(r"      \rowcolor{gray!10}")
            no_str = ""
        else:
            no += 1
            no_str = str(no)
        lines.append(
            f"      {no_str} & {r['centre_en']} & {r['archive_report']} & {n} & {r['age']} & "
            f"{pct(int(r['hpv_pos']), n)} & {pct(int(r['hpv_neg']), n)} & {pct(int(r['hpv_unc']), n)} & "
            f"{pct(int(r['tct_neg']), n)} & {pct(int(r['tct_ascus']), n)} & {pct(int(r['tct_lsil']), n)} & "
            f"{pct(int(r['tct_miss']), n)} & "
            f"{pct(int(r['oct_abnormal_n']), n)} & {pct(int(r['oct_read_missing']), n)} & "
            f"{pct(int(r['cin01']), n)} & {pct(int(r['cin2']), n)} & "
            f"{pct(int(r['cin3']), n)} & {pct(int(r['inv']), n)} \\\\"
        )
    lines += [
        r"      \bottomrule",
        r"    \end{tabular}",
        r"    \end{adjustbox}",
        r"    \vspace{0.35em}",
        r"    \begin{minipage}{\linewidth}",
        r"      \footnotesize",
        r"      \textit{Screened registry cohort from `3000\_nums.xlsx` (MedicalInfo); $N=3{,}010$ rows reduced to $N=3{,}009$ unique OCT examination identifiers. No requirement for colposcopy linkage or released multimodal training paths. OCT second-read abnormal derived from harmonised `OCT二次判读` text (OCTImages sheet preferred). Colposcopy report in archive is centre-level source-data availability (see main text). Histopathology columns are descriptive harmonisation of `病理结果`, `病理级别`, and `后续治疗` text and do not imply independent central pathology review. Ungraded high-grade CIN/HSIL without explicit CIN3/AIS wording was conservatively assigned to the CIN2/HSIL column; explicit CIN2--3 was counted with CIN3+. Invasive cancer required cervical invasive-cancer wording or a cervical-context downstream excision report.}",
        r"    \end{minipage}",
        r"  \end{minipage}",
        r"\end{center}",
    ]
    return "\n".join(lines)


def latex_consort_tikz(c: dict[str, int]) -> str:
    s = c["screened"]
    u = c["unique_registry"]
    dup = c["dup_excluded"]
    oct_a = c["oct_storage_available"]
    ex_oct = c["excluded_oct_missing"]
    final = c["final_multimodal_paired"]
    ex_mm = c["excluded_not_multimodal"]
    ex_col = c["excluded_col_or_alignment"]

    return rf"""% Auto-generated: generate_cohort_supplementary_s1_consort.py
% Requires: \usepackage{{tikz}}
%          \usetikzlibrary{{positioning,shapes.geometric,arrows.meta}}
\begin{{landscape}}
\begin{{center}}
\begin{{tikzpicture}}[
  font=\small,
  node distance=6mm and 10mm,
  stage/.style={{draw, rounded corners=2pt, align=center, text width=44mm, minimum height=14mm, fill=blue!4}},
  exclude/.style={{draw, rounded corners=2pt, align=center, text width=40mm, minimum height=12mm, fill=orange!8}},
  final/.style={{draw, rounded corners=2pt, align=center, text width=48mm, minimum height=14mm, fill=green!10, very thick}},
  arr/.style={{-{{Stealth[length=2.2mm]}}, thick}}
]

\node[stage] (a) {{Administrative registry export\\ (\texttt{{3000\_nums.xlsx}}, MedicalInfo)\\ $N={s:,}$ examinations}};

\node[stage, below=of a] (b) {{Unique OCT examination identifiers\\ (duplicate \texttt{{OCT图像Id}} removed)\\ $N={u:,}$}};

\node[exclude, right=18mm of b] (exdup) {{Excluded duplicate\\ registry rows\\ $n={dup}$}};

\node[stage, below=of b] (c) {{OCT B-scan volumes available on storage\\ (remote Original\_folder mirror)\\ $n={oct_a:,}$}};

\node[exclude, right=18mm of c] (exoct) {{Excluded: OCT volume\\ not locatable on storage\\ $n={ex_oct}$}};

\node[stage, below=of c] (d) {{Paired colposcopy stills aligned by\\ examination date / centre archive rules\\ (candidate linkage universe)}};

\node[exclude, right=18mm of d] (excol) {{Excluded: colposcopy not\\ alignable or no still images\\ $n\approx {ex_col:,}$}};

\node[final, below=of d] (e) {{Final five-centre multimodal analytic cohort\\ OCT + colposcopy + clinical priors\\ (\texttt{{All\_3000\_5cens}})\\ $N={final:,}$ examinations}};

\node[exclude, left=18mm of e] (exrest) {{Excluded from multimodal release\\ (sum of linkage failures)\\ $n={ex_mm}$}};

\draw[arr] (a) -- (b);
\draw[arr] (b) -- (c);
\draw[arr] (c) -- (d);
\draw[arr] (d) -- (e);
\draw[arr, dashed] (b.east) -- (exdup.west);
\draw[arr, dashed] (c.east) -- (exoct.west);
\draw[arr, dashed] (d.east) -- (excol.west);
\draw[arr, dashed] (exrest.east) -- (e.west);

\node[below=8mm of e, align=center, text width=0.92\linewidth] {{
  \footnotesize CONSORT-style flow for the five-centre 3000-examination programme.
  Supplementary Table~S1 characterises all $N={u:,}$ registry rows;
  main Table~characterises the $N={final:,}$ multimodal analytic subset.
}};

\end{{tikzpicture}}
\end{{center}}
\end{{landscape}}
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    reg = load_registry()
    n_unique = len(reg)
    assert n_unique in (3009, 3010), f"Unexpected registry size: {n_unique}"

    tab = build_s1_rows(reg)
    tab.to_csv(OUT_DIR / "supplementary_table_s1_registry_by_center.csv", index=False, encoding="utf-8-sig")
    reg.to_csv(OUT_DIR / "supplementary_table_s1_patient_level.csv", index=False, encoding="utf-8-sig")

    (OUT_DIR / "supplementary_table_s1_registry.tex").write_text(latex_s1_table(tab), encoding="utf-8")

    counts = consort_counts(reg)
    pd.Series(counts).to_csv(OUT_DIR / "consort_flow_counts.csv")
    (FIG_DIR / "cohort_flow_consort_landscape.tex").write_text(latex_consort_tikz(counts), encoding="utf-8")

    print("Wrote:", OUT_DIR / "supplementary_table_s1_registry.tex")
    print("Wrote:", FIG_DIR / "cohort_flow_consort_landscape.tex")
    print("CONSORT counts:", counts)
    print(tab[["centre_en", "n", "age", "oct_abnormal_n"]].to_string())


if __name__ == "__main__":
    main()
