from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

import pandas as pd
import yaml


Backend = Literal["deepface", "prompt_fallback"]


@dataclass(frozen=True)
class PathsConfig:
    prompts_csv: str
    prompts_with_paths_csv: str
    images_dir: str
    audit_results_json: str


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
        prompts_csv=str(paths_raw.get("prompts_csv") or "data/prompts.csv"),
        prompts_with_paths_csv=str(
            paths_raw.get("prompts_with_paths_csv") or "data/prompts_with_paths.csv"
        ),
        images_dir=str(paths_raw.get("images_dir") or "output/images"),
        audit_results_json=str(paths_raw.get("audit_results_json") or "data/audit_results.json"),
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_inputs(paths: PathsConfig) -> pd.DataFrame:
    if os.path.exists(paths.prompts_with_paths_csv):
        df = pd.read_csv(paths.prompts_with_paths_csv, dtype=str, keep_default_na=False)
        if "image_path" not in df.columns:
            raise ValueError(f"Missing image_path column in {paths.prompts_with_paths_csv}")
        return df

    if not os.path.exists(paths.prompts_csv):
        raise FileNotFoundError(
            f"Missing {paths.prompts_csv}. Run prompt_gen.py first."
        )
    base = pd.read_csv(paths.prompts_csv, dtype=str, keep_default_na=False)
    base["image_path"] = ""
    return base


def _try_import_deepface():
    try:
        from deepface import DeepFace
    except Exception:
        return None
    return DeepFace


def _deepface_analyze(deepface: Any, image_path: str) -> dict[str, Any]:
    result = deepface.analyze(
        img_path=image_path,
        actions=["gender", "race"],
        enforce_detection=False,
        silent=True,
    )
    if isinstance(result, list) and result:
        return result[0]
    if isinstance(result, dict):
        return result
    return {}


def _normalize_gender(result: dict[str, Any]) -> tuple[str, float | None]:
    gender = result.get("dominant_gender")
    probs = result.get("gender")
    if isinstance(probs, dict) and gender in probs:
        try:
            return str(gender), float(probs[gender])
        except Exception:
            return str(gender), None
    return str(gender) if gender is not None else "N/A", None


def _normalize_race(result: dict[str, Any]) -> tuple[str, float | None]:
    race = result.get("dominant_race")
    probs = result.get("race")
    if isinstance(probs, dict) and race in probs:
        try:
            return str(race), float(probs[race])
        except Exception:
            return str(race), None
    return str(race) if race is not None else "N/A", None


def _resolve_image_path(*, config_path: str, image_path: str) -> str:
    # Change note: resolve relative CSV paths against config.yaml directory
    # so running `python auditor.py` from a different working directory still works.
    path = os.path.normpath(image_path.strip())
    if not path:
        return ""
    if os.path.exists(path):
        return path
    config_dir = os.path.dirname(os.path.abspath(config_path))
    candidate = os.path.normpath(os.path.join(config_dir, path))
    if os.path.exists(candidate):
        return candidate
    return ""


def main() -> None:
    config_path = "config.yaml"
    paths = load_paths_config(config_path)
    os.makedirs(os.path.dirname(paths.audit_results_json) or ".", exist_ok=True)

    df = _load_inputs(paths)
    deepface = _try_import_deepface()
    backend: Backend = "deepface" if deepface is not None else "prompt_fallback"

    records: list[dict[str, Any]] = []
    total = len(df)
    processed = 0
    skipped = 0

    for _, row in df.iterrows():
        prompt_id = str(row.get("prompt_id", ""))
        image_path = str(row.get("image_path", "")).strip()
        resolved_path = _resolve_image_path(config_path=config_path, image_path=image_path)
        if not resolved_path:
            skipped += 1
            continue

        rec: dict[str, Any] = {
            "prompt_id": prompt_id,
            "category": str(row.get("category", "")),
            "occupation": str(row.get("occupation", "")),
            "race": str(row.get("race", "")) or None,
            "gender": str(row.get("gender", "")) or None,
            "culture": str(row.get("culture", "")) or None,
            "template_id": str(row.get("template_id", "")) or None,
            "replicate": str(row.get("replicate", "")) or None,
            "full_text": str(row.get("full_text", "")),
            "image_path": resolved_path,
            "audited_at": utc_now_iso(),
            "backend": backend,
        }

        if backend == "deepface":
            try:
                result = _deepface_analyze(deepface, image_path=resolved_path)
                pred_gender, gender_conf = _normalize_gender(result)
                pred_race, race_conf = _normalize_race(result)
                rec["predicted_gender"] = pred_gender
                rec["predicted_race"] = pred_race
                rec["gender_confidence"] = gender_conf
                rec["race_confidence"] = race_conf
                face_conf = result.get("face_confidence")
                try:
                    rec["face_confidence"] = float(face_conf) if face_conf is not None else None
                except Exception:
                    rec["face_confidence"] = None
                rec["gender_probs"] = (
                    result.get("gender") if isinstance(result.get("gender"), dict) else None
                )
                rec["race_probs"] = (
                    result.get("race") if isinstance(result.get("race"), dict) else None
                )
            except Exception as e:
                rec["predicted_gender"] = "N/A"
                rec["predicted_race"] = "N/A"
                rec["gender_confidence"] = None
                rec["race_confidence"] = None
                rec["face_confidence"] = None
                rec["gender_probs"] = None
                rec["race_probs"] = None
                rec["error"] = str(e)
        else:
            rec["predicted_gender"] = rec["gender"] or "N/A"
            rec["predicted_race"] = rec["race"] or "N/A"
            rec["gender_confidence"] = None
            rec["race_confidence"] = None
            rec["face_confidence"] = None
            rec["gender_probs"] = None
            rec["race_probs"] = None

        records.append(rec)
        processed += 1
        if processed % 25 == 0:
            print(f"Audited {processed}/{total}")

    with open(paths.audit_results_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Done. Audited {len(records)} images.")
    if skipped:
        print(f"Skipped {skipped} rows due to missing image files.")
    print(f"Wrote -> {paths.audit_results_json}")
    if backend == "prompt_fallback":
        print("DeepFace not available. Install DeepFace to run real face attribute auditing.")


if __name__ == "__main__":
    main()
