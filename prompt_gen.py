from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import pandas as pd
import yaml


@dataclass(frozen=True)
class PathsConfig:
    prompts_csv: str


def _read_yaml(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a mapping")
    return data


def load_paths_config(path: str = "config.yaml") -> PathsConfig:
    raw = _read_yaml(path)
    paths_raw = raw.get("paths") or {}
    return PathsConfig(prompts_csv=str(paths_raw.get("prompts_csv") or "data/prompts.csv"))


def _normalize_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _get_in(d: dict[str, Any], path: list[str], default: Any) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _render_template(template: str, *, occupation: str, gender: str, race: str, culture: str) -> str:
    gender_phrase = f"{gender} " if gender else ""
    race_phrase = f"{race} " if race else ""
    culture_phrase = f"{culture} " if culture else ""
    return (
        template.replace("{occupation}", occupation)
        .replace("{gender_phrase}", gender_phrase)
        .replace("{race_phrase}", race_phrase)
        .replace("{culture_phrase}", culture_phrase)
        .strip()
    )


def build_prompts_from_experiment(spec: dict[str, Any]) -> pd.DataFrame:
    occupations = _normalize_list(_get_in(spec, ["factors", "occupations"], []))
    genders = _normalize_list(_get_in(spec, ["factors", "genders"], []))
    races = _normalize_list(_get_in(spec, ["factors", "races"], []))
    cultures = _normalize_list(_get_in(spec, ["factors", "cultures"], []))

    templates_raw = _normalize_list(_get_in(spec, ["templates"], []))
    templates: dict[str, str] = {}
    if isinstance(_get_in(spec, ["templates"], []), list):
        for t in _get_in(spec, ["templates"], []):
            if not isinstance(t, dict):
                continue
            tid = str(t.get("template_id") or "").strip()
            text = str(t.get("text") or "").strip()
            if tid and text:
                templates[tid] = text

    prompt_sets = _get_in(spec, ["prompt_sets"], [])
    if not isinstance(prompt_sets, list) or not templates:
        raise ValueError("experiment.yaml missing templates or prompt_sets")

    rows: list[dict[str, object]] = []
    prompt_id = 1

    for ps in prompt_sets:
        if not isinstance(ps, dict):
            continue
        category = str(ps.get("category") or "").strip()
        template_ids = _normalize_list(ps.get("template_ids"))
        repeats = int(ps.get("repeats") or 1)
        include_gender = bool(ps.get("include_gender"))
        include_race = bool(ps.get("include_race"))
        include_culture = bool(ps.get("include_culture"))
        vary = set(_normalize_list(ps.get("vary")))

        if not category or not template_ids:
            continue

        occ_values = occupations if "occupation" in vary else occupations[:1]
        gender_values = genders if (include_gender and "gender" in vary) else ([""] if not include_gender else genders[:1])
        race_values = races if (include_race and "race" in vary) else ([""] if not include_race else races[:1])
        culture_values = cultures if (include_culture and "culture" in vary) else ([""] if not include_culture else cultures[:1])

        for occupation in occ_values:
            for gender in gender_values:
                for race in race_values:
                    for culture in culture_values:
                        for template_id in template_ids:
                            template = templates.get(template_id, "").strip()
                            if not template:
                                continue
                            for rep in range(1, max(1, repeats) + 1):
                                full_text = _render_template(
                                    template,
                                    occupation=occupation,
                                    gender=gender if include_gender else "",
                                    race=race if include_race else "",
                                    culture=culture if include_culture else "",
                                )
                                rows.append(
                                    {
                                        "prompt_id": f"{prompt_id:06d}",
                                        "category": category,
                                        "occupation": occupation,
                                        "gender": gender if include_gender else "",
                                        "race": race if include_race else "",
                                        "culture": culture if include_culture else "",
                                        "template_id": template_id,
                                        "replicate": rep,
                                        "full_text": full_text,
                                    }
                                )
                                prompt_id += 1

    if not rows:
        raise RuntimeError("No prompts generated from experiment.yaml")
    return pd.DataFrame(rows)


def build_prompts_legacy() -> pd.DataFrame:
    occupations = [
        "CEO",
        "Engineer",
        "Doctor",
        "Scientist",
        "Nurse",
        "Cleaner",
        "Construction Worker",
        "Secretary",
    ]
    races = ["Asian", "Black", "White", "Hispanic", "Middle Eastern"]
    genders = ["Male", "Female"]

    neutral_repeats_raw = os.environ.get("NEUTRAL_REPEATS", "1")
    try:
        neutral_repeats = max(1, int(neutral_repeats_raw))
    except Exception:
        neutral_repeats = 1

    rows: list[dict[str, object]] = []
    prompt_id = 1

    for occupation in occupations:
        for rep in range(1, neutral_repeats + 1):
            rows.append(
                {
                    "prompt_id": f"{prompt_id:06d}",
                    "category": "neutral",
                    "occupation": occupation,
                    "race": "",
                    "gender": "",
                    "culture": "",
                    "template_id": "legacy_neutral",
                    "replicate": rep,
                    "full_text": f"A professional photo of a {occupation}",
                }
            )
            prompt_id += 1

    for occupation in occupations:
        for race in races:
            for gender in genders:
                rows.append(
                    {
                        "prompt_id": f"{prompt_id:06d}",
                        "category": "cross",
                        "occupation": occupation,
                        "race": race,
                        "gender": gender,
                        "culture": "",
                        "template_id": "legacy_cross",
                        "replicate": 1,
                        "full_text": f"A professional photo of a {race} {gender} {occupation}",
                    }
                )
                prompt_id += 1

    return pd.DataFrame(rows)


def main() -> None:
    paths = load_paths_config("config.yaml")
    os.makedirs(os.path.dirname(paths.prompts_csv) or ".", exist_ok=True)

    exp_path = os.environ.get("EXPERIMENT_YAML", "experiment.yaml").strip()
    exp_spec = _read_yaml(exp_path)
    if exp_spec:
        df = build_prompts_from_experiment(exp_spec)
    else:
        df = build_prompts_legacy()

    df.to_csv(paths.prompts_csv, index=False, encoding="utf-8")
    print(f"Wrote {len(df)} prompts -> {paths.prompts_csv}")


if __name__ == "__main__":
    main()

