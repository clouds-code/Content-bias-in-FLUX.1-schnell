from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from bias_metrics import adherence_rate, cramer_v, js_divergence


@dataclass(frozen=True)
class PathsConfig:
    audit_results_json: str
    reports_dir: str


def _read_yaml(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing config file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("config.yaml must be a mapping")
    return data


def load_paths_config(path: str = "config.yaml") -> PathsConfig:
    raw = _read_yaml(path)
    paths_raw = raw.get("paths") or {}
    return PathsConfig(
        audit_results_json=str(paths_raw.get("audit_results_json") or "data/audit_results.json"),
        reports_dir=str(paths_raw.get("reports_dir") or "reports"),
    )


def _load_results(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Run auditor.py first.")
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    return pd.DataFrame.from_records(records)


def _normalize_text(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip()


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series([""] * len(df), index=df.index)


def _normalize_label(value: str) -> str:
    v = (value or "").strip().lower()
    if v in {"man"}:
        return "male"
    if v in {"woman"}:
        return "female"
    if v in {"latino", "hispanic", "latino_hispanic", "latino hispanic"}:
        return "latino hispanic"
    if v in {"middle_eastern", "middle eastern"}:
        return "middle eastern"
    return v


def _safe_probs_from_counts(counts: pd.Series) -> np.ndarray:
    arr = counts.to_numpy(dtype=float)
    s = float(arr.sum())
    if s <= 0:
        return np.zeros_like(arr, dtype=float)
    return arr / s


def main() -> None:
    na_value = "n/a"
    paths = load_paths_config("config.yaml")
    os.makedirs(paths.reports_dir, exist_ok=True)

    df = _load_results(paths.audit_results_json)
    if df.empty:
        raise RuntimeError("audit_results.json is empty.")

    df["category"] = _normalize_text(_col(df, "category"))
    df["occupation"] = _normalize_text(_col(df, "occupation"))
    df["predicted_gender"] = _normalize_text(_col(df, "predicted_gender")).map(_normalize_label)
    df["predicted_race"] = _normalize_text(_col(df, "predicted_race")).map(_normalize_label)
    df["gender"] = _normalize_text(_col(df, "gender")).map(_normalize_label)
    df["race"] = _normalize_text(_col(df, "race")).map(_normalize_label)
    df["culture"] = _normalize_text(_col(df, "culture"))
    df["backend"] = _normalize_text(_col(df, "backend"))

    df_real = df[df["backend"].eq("deepface")].copy()
    if df_real.empty:
        df_real = df.copy()

    neutral = df_real[df_real["category"].eq("neutral")].copy()
    if neutral.empty:
        neutral = df_real.copy()

    # Change note: use groupby+unstack instead of pivot_table to keep integer counts.
    gender_pivot = (
        neutral[neutral["predicted_gender"].ne(na_value)]
        .groupby(["occupation", "predicted_gender"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    gender_prop = gender_pivot.div(gender_pivot.sum(axis=1), axis=0).fillna(0)

    focus_jobs = ["Nurse", "CEO"]
    focus = gender_prop.loc[gender_prop.index.intersection(focus_jobs)].copy()
    if not focus.empty:
        ax = focus.plot(kind="bar", stacked=True, figsize=(8, 4))
        ax.set_title("Gender distribution (neutral prompts)")
        ax.set_xlabel("Occupation")
        ax.set_ylabel("Proportion")
        plt.tight_layout()
        out_path = os.path.join(paths.reports_dir, "gender_nurse_vs_ceo.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Wrote -> {out_path}")

        # Save table
        out_csv = os.path.join(paths.reports_dir, "neutral_gender_dist.csv")
        gender_pivot.to_csv(out_csv, encoding="utf-8")
        print(f"Wrote -> {out_csv}")

    race_pivot = (
        neutral[neutral["predicted_race"].ne(na_value)]
        .groupby(["occupation", "predicted_race"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    race_prop = race_pivot.div(race_pivot.sum(axis=1), axis=0).fillna(0)

    ax = race_prop.plot(kind="bar", stacked=True, figsize=(12, 5))
    n_by_occ = race_pivot.sum(axis=1)
    ax.set_title(
        "Race distribution (neutral prompts, n per occupation: "
        f"{int(n_by_occ.min())}-{int(n_by_occ.max())})"
    )
    ax.set_xlabel("Occupation")
    ax.set_ylabel("Proportion")
    plt.tight_layout()
    out_path = os.path.join(paths.reports_dir, "race_distribution_stacked.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Wrote -> {out_path}")

    # Save table
    out_csv = os.path.join(paths.reports_dir, "neutral_race_dist.csv")
    race_pivot.to_csv(out_csv, encoding="utf-8")
    print(f"Wrote -> {out_csv}")

    metrics: dict[str, Any] = {}
    metrics["counts"] = {
        "total_images": int(len(df)),
        "audited_images": int(len(df_real)),
        "neutral_images": int(len(neutral)),
    }
    if not race_pivot.empty:
        overall_counts = race_pivot.sum(axis=0)
        overall_probs = _safe_probs_from_counts(overall_counts)
        js_by_occ: dict[str, float] = {}
        for occ in race_pivot.index:
            p = _safe_probs_from_counts(race_pivot.loc[occ])
            js_by_occ[str(occ)] = float(js_divergence(p, overall_probs))
        metrics["js_divergence_race_neutral_vs_overall"] = js_by_occ

    cf = df_real[df_real["category"].str.startswith("cf_")].copy()
    if not cf.empty:
        cf_gender = cf[cf["gender"].ne("") & cf["predicted_gender"].ne(na_value)]
        if not cf_gender.empty:
            metrics["gender_adherence_accuracy"] = float(
                (cf_gender["gender"] == cf_gender["predicted_gender"]).mean()
            )
        cf_race = cf[cf["race"].ne("") & cf["predicted_race"].ne(na_value)]
        if not cf_race.empty:
            metrics["race_adherence_accuracy"] = float(
                (cf_race["race"] == cf_race["predicted_race"]).mean()
            )

    cross = df_real[
        df_real["category"].str.startswith("cf_") | df_real["category"].eq("cross")
    ].copy()
    if not cross.empty:
        cross_gender = adherence_rate(
            specified=cross["gender"], predicted=cross["predicted_gender"], na_value=na_value
        )
        cross_race = adherence_rate(
            specified=cross["race"], predicted=cross["predicted_race"], na_value=na_value
        )
        metrics["cross_gender_adherence"] = {
            "n": int(cross_gender.n),
            "accuracy": (
                float(cross_gender.accuracy)
                if not np.isnan(cross_gender.accuracy)
                else None
            ),
        }
        metrics["cross_race_adherence"] = {
            "n": int(cross_race.n),
            "accuracy": (
                float(cross_race.accuracy)
                if not np.isnan(cross_race.accuracy)
                else None
            ),
        }

        by_occ: dict[str, Any] = {}
        for occ, g in cross.groupby("occupation"):
            gr = adherence_rate(
                specified=g["race"], predicted=g["predicted_race"], na_value=na_value
            )
            gg = adherence_rate(
                specified=g["gender"], predicted=g["predicted_gender"], na_value=na_value
            )
            by_occ[str(occ)] = {
                "n": int(len(g)),
                "gender_adherence_accuracy": (
                    float(gg.accuracy) if not np.isnan(gg.accuracy) else None
                ),
                "gender_adherence_n": int(gg.n),
                "race_adherence_accuracy": (
                    float(gr.accuracy) if not np.isnan(gr.accuracy) else None
                ),
                "race_adherence_n": int(gr.n),
            }
        metrics["cross_adherence_by_occupation"] = by_occ

        cross_race_valid = cross[cross["race"].ne("")]
        race_conf = pd.crosstab(cross_race_valid["race"], cross_race_valid["predicted_race"])
        if not race_conf.empty:
            plt.figure(figsize=(10, 4))
            sns.heatmap(
                race_conf.div(race_conf.sum(axis=1), axis=0).fillna(0),
                cmap="mako",
                linewidths=0.5,
            )
            plt.title("Race label adherence (cross prompts)")
            plt.xlabel("Predicted race")
            plt.ylabel("Specified race")
            plt.tight_layout()
            out_path = os.path.join(paths.reports_dir, "cross_race_confusion_heatmap.png")
            plt.savefig(out_path, dpi=200)
            plt.close()
            print(f"Wrote -> {out_path}")

        cross_gender_valid = cross[cross["gender"].ne("")]
        gender_conf = pd.crosstab(
            cross_gender_valid["gender"],
            cross_gender_valid["predicted_gender"],
        )
        if not gender_conf.empty:
            plt.figure(figsize=(6, 3))
            sns.heatmap(
                gender_conf.div(gender_conf.sum(axis=1), axis=0).fillna(0),
                cmap="mako",
                linewidths=0.5,
                annot=True,
                fmt=".2f",
            )
            plt.title("Gender label adherence (cross prompts)")
            plt.xlabel("Predicted gender")
            plt.ylabel("Specified gender")
            plt.tight_layout()
            out_path = os.path.join(
                paths.reports_dir, "cross_gender_confusion_heatmap.png"
            )
            plt.savefig(out_path, dpi=200)
            plt.close()
            print(f"Wrote -> {out_path}")

    cf_culture = df_real[df_real["category"].eq("cf_culture")].copy()
    if not cf_culture.empty:
        tab = pd.crosstab(cf_culture["culture"], cf_culture["predicted_race"])
        metrics["culture_to_race_cramers_v"] = float(cramer_v(tab))
        plt.figure(figsize=(10, 4))
        sns.heatmap(
            tab.div(tab.sum(axis=1), axis=0).fillna(0),
            cmap="mako",
            linewidths=0.5,
        )
        plt.title("Predicted race distribution by culture (cf_culture)")
        plt.xlabel("Predicted race")
        plt.ylabel("Culture")
        plt.tight_layout()
        out_path = os.path.join(paths.reports_dir, "culture_to_race_heatmap.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Wrote -> {out_path}")

    # Change note: write summary after all metrics are computed so cross fields are present.
    summary_rows: list[dict[str, Any]] = []
    for occ in sorted(df_real["occupation"].unique()):
        if not occ:
            continue
        row: dict[str, Any] = {"occupation": occ}
        js_map = metrics.get("js_divergence_race_neutral_vs_overall") or {}
        if js_map:
            row["js_divergence"] = js_map.get(occ, "")

        cross_by_occ = metrics.get("cross_adherence_by_occupation") or {}
        if cross_by_occ:
            cross_occ = (
                cross_by_occ.get(occ, {}) if isinstance(cross_by_occ, dict) else {}
            )
            row["cross_gender_acc"] = cross_occ.get("gender_adherence_accuracy", "")
            row["cross_race_acc"] = cross_occ.get("race_adherence_accuracy", "")

        summary_rows.append(row)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        out_csv = os.path.join(paths.reports_dir, "bias_metrics_summary.csv")
        summary_df.to_csv(out_csv, index=False, encoding="utf-8")
        print(f"Wrote -> {out_csv}")

    out_json = os.path.join(paths.reports_dir, "bias_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Wrote -> {out_json}")

    if "race" in df_real.columns:
        with_race_prompt = df_real[df_real["race"].ne("")].copy()
        if not with_race_prompt.empty:
            with_race_prompt["race_confidence"] = pd.to_numeric(
                with_race_prompt.get("race_confidence"), errors="coerce"
            )
            heat = (
                with_race_prompt.pivot_table(
                    index="occupation",
                    columns="race",
                    values="race_confidence",
                    aggfunc="mean",
                )
                .sort_index()
                .fillna(0)
            )
            plt.figure(figsize=(10, 5))
            sns.heatmap(heat, cmap="viridis", linewidths=0.5)
            plt.title("Mean race confidence by prompt race label")
            plt.xlabel("Prompt race label")
            plt.ylabel("Occupation")
            plt.tight_layout()
            out_path = os.path.join(paths.reports_dir, "race_confidence_heatmap.png")
            plt.savefig(out_path, dpi=200)
            plt.close()
            print(f"Wrote -> {out_path}")

    report_lines: list[str] = []
    report_lines.append("Audit summary:")
    report_lines.append(f"- total_images: {len(df)}")
    report_lines.append(
        f"- neutral_images: {len(neutral)} (fallback to all if none)"
    )
    if not focus.empty:
        report_lines.append("- nurse_vs_ceo_gender:")
        for occ in focus.index:
            row = focus.loc[occ]
            parts = ", ".join([f"{col}={row[col]:.2f}" for col in focus.columns])
            report_lines.append(f"  - {occ}: {parts}")

    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
