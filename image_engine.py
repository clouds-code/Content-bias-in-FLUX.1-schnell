from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import yaml


@dataclass(frozen=True)
class OpenAIConfig:
    api_key_env: str
    model: str
    size: str
    quality: str
    response_format: str
    max_retries: int
    initial_backoff_seconds: float
    max_backoff_seconds: float


@dataclass(frozen=True)
class GeneratorConfig:
    backend: str


@dataclass(frozen=True)
class DiffusersConfig:
    model_id: str
    device: str
    seed: int
    num_inference_steps: int
    guidance_scale: float
    width: int
    height: int


@dataclass(frozen=True)
class HuggingFaceConfig:
    token_env: str
    api_base: str
    model_id: str
    timeout_seconds: float
    max_retries: int
    initial_backoff_seconds: float
    max_backoff_seconds: float


@dataclass(frozen=True)
class SiliconFlowConfig:
    api_key_env: str
    base_url: str
    model: str
    size: str
    response_format: str
    timeout_seconds: float
    max_retries: int
    initial_backoff_seconds: float
    max_backoff_seconds: float


@dataclass(frozen=True)
class DummyConfig:
    width: int
    height: int
    font_size: int


@dataclass(frozen=True)
class PathsConfig:
    prompts_csv: str
    prompts_with_paths_csv: str
    images_dir: str


@dataclass(frozen=True)
class RunConfig:
    limit: int | None


@dataclass(frozen=True)
class Config:
    generator: GeneratorConfig
    openai: OpenAIConfig
    diffusers: DiffusersConfig
    huggingface: HuggingFaceConfig
    siliconflow: SiliconFlowConfig
    dummy: DummyConfig
    paths: PathsConfig
    run: RunConfig


def _read_yaml(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing config file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("config.yaml must be a mapping")
    return data


def load_config(path: str = "config.yaml") -> Config:
    raw = _read_yaml(path)
    generator_raw = raw.get("generator") or {}
    openai_raw = raw.get("openai") or {}
    diffusers_raw = raw.get("diffusers") or {}
    hf_raw = raw.get("huggingface") or {}
    sf_raw = raw.get("siliconflow") or {}
    dummy_raw = raw.get("dummy") or {}
    paths_raw = raw.get("paths") or {}
    run_raw = raw.get("run") or {}

    generator_cfg = GeneratorConfig(
        backend=str(generator_raw.get("backend") or "openai").strip().lower()
    )
    openai_cfg = OpenAIConfig(
        api_key_env=str(openai_raw.get("api_key_env") or "OPENAI_API_KEY"),
        model=str(openai_raw.get("model") or "dall-e-3"),
        size=str(openai_raw.get("size") or "1024x1024"),
        quality=str(openai_raw.get("quality") or "standard"),
        response_format=str(openai_raw.get("response_format") or "b64_json"),
        max_retries=int(openai_raw.get("max_retries") or 6),
        initial_backoff_seconds=float(openai_raw.get("initial_backoff_seconds") or 2),
        max_backoff_seconds=float(openai_raw.get("max_backoff_seconds") or 60),
    )
    diffusers_cfg = DiffusersConfig(
        model_id=str(diffusers_raw.get("model_id") or "runwayml/stable-diffusion-v1-5"),
        device=str(diffusers_raw.get("device") or "auto").strip().lower(),
        seed=int(diffusers_raw.get("seed") or 42),
        num_inference_steps=int(diffusers_raw.get("num_inference_steps") or 30),
        guidance_scale=float(diffusers_raw.get("guidance_scale") or 7.5),
        width=int(diffusers_raw.get("width") or 512),
        height=int(diffusers_raw.get("height") or 512),
    )
    hf_cfg = HuggingFaceConfig(
        token_env=str(hf_raw.get("token_env") or "HF_TOKEN"),
        api_base=str(hf_raw.get("api_base") or "https://api-inference.huggingface.co"),
        model_id=str(hf_raw.get("model_id") or "black-forest-labs/FLUX.1-schnell"),
        timeout_seconds=float(hf_raw.get("timeout_seconds") or 120),
        max_retries=int(hf_raw.get("max_retries") or 8),
        initial_backoff_seconds=float(hf_raw.get("initial_backoff_seconds") or 2),
        max_backoff_seconds=float(hf_raw.get("max_backoff_seconds") or 60),
    )
    sf_cfg = SiliconFlowConfig(
        api_key_env=str(sf_raw.get("api_key_env") or "SILICONFLOW_API_KEY"),
        base_url=str(sf_raw.get("base_url") or "https://api.siliconflow.cn/v1"),
        model=str(sf_raw.get("model") or "black-forest-labs/FLUX.1-schnell"),
        size=str(sf_raw.get("size") or "1024x1024"),
        response_format=str(sf_raw.get("response_format") or "b64_json"),
        timeout_seconds=float(sf_raw.get("timeout_seconds") or 180),
        max_retries=int(sf_raw.get("max_retries") or 8),
        initial_backoff_seconds=float(sf_raw.get("initial_backoff_seconds") or 2),
        max_backoff_seconds=float(sf_raw.get("max_backoff_seconds") or 60),
    )
    dummy_cfg = DummyConfig(
        width=int(dummy_raw.get("width") or 1024),
        height=int(dummy_raw.get("height") or 1024),
        font_size=int(dummy_raw.get("font_size") or 20),
    )
    paths_cfg = PathsConfig(
        prompts_csv=str(paths_raw.get("prompts_csv") or "data/prompts.csv"),
        prompts_with_paths_csv=str(
            paths_raw.get("prompts_with_paths_csv") or "data/prompts_with_paths.csv"
        ),
        images_dir=str(paths_raw.get("images_dir") or "output/images"),
    )
    limit_value = run_raw.get("limit", None)
    if limit_value is None:
        limit = None
    else:
        limit = int(limit_value)
    return Config(
        generator=generator_cfg,
        openai=openai_cfg,
        diffusers=diffusers_cfg,
        huggingface=hf_cfg,
        siliconflow=sf_cfg,
        dummy=dummy_cfg,
        paths=paths_cfg,
        run=RunConfig(limit=limit),
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _sanitize_filename(value: str) -> str:
    keep = []
    for ch in value:
        if ch.isalnum() or ch in {" ", "-", "_"}:
            keep.append(ch)
    out = "".join(keep).strip().replace(" ", "_")
    return out or "unknown"


def _decode_b64_to_bytes(b64_str: str) -> bytes:
    return base64.b64decode(b64_str.encode("utf-8"))


def _openai_client(api_key: str):
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError(
            "Missing dependency 'openai'. Install requirements.txt first."
        ) from e
    return OpenAI(api_key=api_key)


def _hf_generate_image_bytes(
    *,
    token: str,
    api_base: str,
    model_id: str,
    prompt: str,
    timeout_seconds: float,
) -> bytes:
    try:
        import requests
    except Exception as e:
        raise RuntimeError("Missing dependency 'requests'. Install requirements.txt first.") from e

    base = api_base.rstrip("/")
    url = f"{base}/models/{model_id}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "image/png",
    }
    payload = {
        "inputs": prompt,
        "options": {"wait_for_model": True},
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout_seconds)
    if resp.status_code == 200:
        content_type = (resp.headers.get("content-type") or "").lower()
        if content_type.startswith("image/"):
            return resp.content

    try:
        body = resp.json()
    except Exception:
        body = resp.text

    raise RuntimeError(f"HuggingFace API {resp.status_code}: {body}")


def _siliconflow_generate_image_bytes(
    *,
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    size: str,
    response_format: str,
    timeout_seconds: float,
) -> bytes:
    try:
        import requests
    except Exception as e:
        raise RuntimeError("Missing dependency 'requests'. Install requirements.txt first.") from e

    url = base_url.rstrip("/") + "/images/generations"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "model": model,
        "prompt": prompt,
        "size": size,
        "n": 1,
        "response_format": response_format,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout_seconds)

    content_type = (resp.headers.get("content-type") or "").lower()
    if resp.status_code == 200 and content_type.startswith("image/"):
        return resp.content

    try:
        body = resp.json()
    except Exception:
        raise RuntimeError(f"SiliconFlow API {resp.status_code}: {resp.text}") from None

    if resp.status_code != 200:
        raise RuntimeError(f"SiliconFlow API {resp.status_code}: {body}")

    data = body.get("data")
    if isinstance(data, list) and data:
        first = data[0] if isinstance(data[0], dict) else None
        if isinstance(first, dict):
            b64_json = first.get("b64_json")
            if isinstance(b64_json, str) and b64_json:
                return _decode_b64_to_bytes(b64_json)
            url_value = first.get("url")
            if isinstance(url_value, str) and url_value:
                dl = requests.get(url_value, timeout=timeout_seconds)
                dl.raise_for_status()
                return dl.content

    raise RuntimeError(f"SiliconFlow API 200: unexpected response: {body}")


def generate_image_b64(
    client: Any,
    *,
    model: str,
    prompt: str,
    size: str,
    quality: str,
    response_format: str,
) -> str:
    resp = client.images.generate(
        model=model,
        prompt=prompt,
        size=size,
        quality=quality,
        response_format=response_format,
        n=1,
    )
    data0 = resp.data[0]
    if getattr(data0, "b64_json", None):
        return data0.b64_json
    raise RuntimeError("OpenAI response did not include b64_json")


def _dummy_image_bytes(
    *,
    width: int,
    height: int,
    font_size: int,
    text: str,
) -> bytes:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as e:
        raise RuntimeError("Missing dependency 'pillow'. Install requirements.txt first.") from e

    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    max_chars = 60
    lines = []
    remaining = text.strip()
    while remaining:
        lines.append(remaining[:max_chars])
        remaining = remaining[max_chars:]
        if len(lines) >= 20:
            break

    y = 20
    for line in lines:
        draw.text((20, y), line, fill=(0, 0, 0), font=font)
        y += font_size + 6

    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _diffusers_pipeline(cfg: DiffusersConfig):
    try:
        import torch
    except Exception as e:
        raise RuntimeError(
            "Missing dependencies for diffusers backend. "
            "Install: pip install diffusers transformers accelerate torch"
        ) from e

    if cfg.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = cfg.device

    dtype = torch.float16 if device == "cuda" else torch.float32
    try:
        from diffusers import StableDiffusionPipeline

        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                cfg.model_id,
                torch_dtype=dtype,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
            )
        except TypeError:
            pipe = StableDiffusionPipeline.from_pretrained(
                cfg.model_id,
                torch_dtype=dtype,
                safety_checker=None,
                feature_extractor=None,
            )
    except Exception:
        try:
            from diffusers import DiffusionPipeline
        except Exception as e:
            raise RuntimeError(
                "Failed to import diffusers pipelines. "
                "This is often caused by diffusers/transformers version mismatch. "
                "Try: pip install -U diffusers transformers"
            ) from e
        try:
            pipe = DiffusionPipeline.from_pretrained(
                cfg.model_id,
                torch_dtype=dtype,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
            )
        except TypeError:
            pipe = DiffusionPipeline.from_pretrained(
                cfg.model_id,
                torch_dtype=dtype,
                safety_checker=None,
                feature_extractor=None,
            )
    pipe = pipe.to(device)
    try:
        pipe.safety_checker = None
    except Exception:
        pass
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass
    if device == "cpu":
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass
    return pipe, device, torch


def _should_retry(exc: Exception) -> bool:
    msg = str(exc)
    retry_markers = [
        "HuggingFace API 429",
        "HuggingFace API 500",
        "HuggingFace API 502",
        "HuggingFace API 503",
        "HuggingFace API 504",
        "429",
        "Rate limit",
        "rate limit",
        "timeout",
        "timed out",
        "temporarily",
        "loading",
        "is currently loading",
        "服务异常，请稍后重试",
        "Internal Server Error",
        "502",
        "503",
        "504",
    ]
    return any(m in msg for m in retry_markers)


def _backoff_seconds(attempt: int, initial: float, max_backoff: float) -> float:
    return min(max_backoff, initial * (2 ** max(0, attempt - 1)))


def _existing_or_empty_dataframe(path: str) -> pd.DataFrame | None:
    if os.path.exists(path):
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    return None


def _load_prompts(paths: PathsConfig) -> pd.DataFrame:
    if not os.path.exists(paths.prompts_csv):
        raise FileNotFoundError(
            f"Missing {paths.prompts_csv}. Run prompt_gen.py first."
        )
    base = pd.read_csv(paths.prompts_csv, dtype=str, keep_default_na=False)
    existing = _existing_or_empty_dataframe(paths.prompts_with_paths_csv)
    if existing is None:
        out = base.copy()
        if "image_path" not in out.columns:
            out["image_path"] = ""
        if "generated_at" not in out.columns:
            out["generated_at"] = ""
        if "error" not in out.columns:
            out["error"] = ""
        return out

    merged = base.merge(
        existing[["prompt_id", "image_path", "generated_at", "error"]],
        on="prompt_id",
        how="left",
    )
    for col in ["image_path", "generated_at", "error"]:
        if col not in merged.columns:
            merged[col] = ""
        merged[col] = merged[col].fillna("")
    return merged


def main() -> None:
    cfg = load_config("config.yaml")
    ensure_dir(os.path.dirname(cfg.paths.prompts_with_paths_csv) or ".")
    ensure_dir(cfg.paths.images_dir)

    df = _load_prompts(cfg.paths)
    if cfg.run.limit is not None:
        df = df.head(cfg.run.limit).copy()

    backend = cfg.generator.backend
    if backend == "openai":
        api_key = os.environ.get(cfg.openai.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(
                f"Missing API key env var {cfg.openai.api_key_env}. "
                "Set it or switch config.yaml generator.backend to diffusers/dummy."
            )
        client = _openai_client(api_key)
    elif backend == "huggingface":
        token = os.environ.get(cfg.huggingface.token_env, "").strip()
        if not token:
            raise RuntimeError(
                f"Missing Hugging Face token env var {cfg.huggingface.token_env}. "
                "Set it or switch config.yaml generator.backend to diffusers/dummy."
            )
        client = token
    elif backend == "siliconflow":
        api_key = os.environ.get(cfg.siliconflow.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(
                f"Missing SiliconFlow API key env var {cfg.siliconflow.api_key_env}. "
                "Set it or switch config.yaml generator.backend."
            )
        client = api_key
    elif backend == "diffusers":
        pipe, device, torch = _diffusers_pipeline(cfg.diffusers)
        client = (pipe, device, torch)
    elif backend == "dummy":
        client = None
    else:
        raise ValueError(
            "generator.backend must be one of: openai, huggingface, siliconflow, diffusers, dummy"
        )

    total = len(df)
    generated = 0
    for i, row in df.iterrows():
        prompt_id = str(row["prompt_id"])
        occupation = str(row.get("occupation", "unknown"))
        prompt_text = str(row["full_text"])
        existing_path = str(row.get("image_path", "")).strip()

        occ_dir = os.path.join(cfg.paths.images_dir, _sanitize_filename(occupation))
        ensure_dir(occ_dir)
        filename = f"{prompt_id}_{_sanitize_filename(occupation)}.png"
        target_path = os.path.join(occ_dir, filename)

        if existing_path and os.path.exists(existing_path):
            df.at[i, "image_path"] = existing_path
            df.at[i, "error"] = ""
            generated += 1
            if generated % 10 == 0:
                print(f"Progress {generated}/{total}")
            continue

        if os.path.exists(target_path):
            df.at[i, "image_path"] = target_path
            df.at[i, "generated_at"] = df.at[i, "generated_at"] or utc_now_iso()
            df.at[i, "error"] = ""
            generated += 1
            if generated % 10 == 0:
                print(f"Progress {generated}/{total}")
            continue

        print(f"Generating {generated + 1}/{total} prompt_id={prompt_id}")
        last_error: str = ""
        if backend == "openai":
            for attempt in range(1, cfg.openai.max_retries + 1):
                try:
                    b64_json = generate_image_b64(
                        client,
                        model=cfg.openai.model,
                        prompt=prompt_text,
                        size=cfg.openai.size,
                        quality=cfg.openai.quality,
                        response_format=cfg.openai.response_format,
                    )
                    img_bytes = _decode_b64_to_bytes(b64_json)
                    with open(target_path, "wb") as f:
                        f.write(img_bytes)

                    df.at[i, "image_path"] = target_path
                    df.at[i, "generated_at"] = utc_now_iso()
                    df.at[i, "error"] = ""
                    generated += 1
                    break
                except Exception as e:
                    last_error = str(e)
                    if attempt >= cfg.openai.max_retries or not _should_retry(e):
                        df.at[i, "image_path"] = ""
                        df.at[i, "generated_at"] = utc_now_iso()
                        df.at[i, "error"] = last_error
                        print(f"Failed prompt_id={prompt_id}: {last_error}")
                        break

                    sleep_s = _backoff_seconds(
                        attempt,
                        cfg.openai.initial_backoff_seconds,
                        cfg.openai.max_backoff_seconds,
                    )
                    print(
                        f"Retry {attempt}/{cfg.openai.max_retries} after {sleep_s}s: {last_error}"
                    )
                    time.sleep(sleep_s)
        elif backend == "huggingface":
            for attempt in range(1, cfg.huggingface.max_retries + 1):
                try:
                    img_bytes = _hf_generate_image_bytes(
                        token=str(client),
                        api_base=cfg.huggingface.api_base,
                        model_id=cfg.huggingface.model_id,
                        prompt=prompt_text,
                        timeout_seconds=cfg.huggingface.timeout_seconds,
                    )
                    with open(target_path, "wb") as f:
                        f.write(img_bytes)
                    df.at[i, "image_path"] = target_path
                    df.at[i, "generated_at"] = utc_now_iso()
                    df.at[i, "error"] = ""
                    generated += 1
                    break
                except Exception as e:
                    last_error = str(e)
                    if attempt >= cfg.huggingface.max_retries or not _should_retry(e):
                        df.at[i, "image_path"] = ""
                        df.at[i, "generated_at"] = utc_now_iso()
                        df.at[i, "error"] = last_error
                        print(f"Failed prompt_id={prompt_id}: {last_error}")
                        break
                    sleep_s = _backoff_seconds(
                        attempt,
                        cfg.huggingface.initial_backoff_seconds,
                        cfg.huggingface.max_backoff_seconds,
                    )
                    print(
                        f"Retry {attempt}/{cfg.huggingface.max_retries} after {sleep_s}s: {last_error}"
                    )
                    time.sleep(sleep_s)
        elif backend == "siliconflow":
            for attempt in range(1, cfg.siliconflow.max_retries + 1):
                try:
                    img_bytes = _siliconflow_generate_image_bytes(
                        api_key=str(client),
                        base_url=cfg.siliconflow.base_url,
                        model=cfg.siliconflow.model,
                        prompt=prompt_text,
                        size=cfg.siliconflow.size,
                        response_format=cfg.siliconflow.response_format,
                        timeout_seconds=cfg.siliconflow.timeout_seconds,
                    )
                    with open(target_path, "wb") as f:
                        f.write(img_bytes)
                    df.at[i, "image_path"] = target_path
                    df.at[i, "generated_at"] = utc_now_iso()
                    df.at[i, "error"] = ""
                    generated += 1
                    break
                except Exception as e:
                    last_error = str(e)
                    if attempt >= cfg.siliconflow.max_retries or not _should_retry(e):
                        df.at[i, "image_path"] = ""
                        df.at[i, "generated_at"] = utc_now_iso()
                        df.at[i, "error"] = last_error
                        print(f"Failed prompt_id={prompt_id}: {last_error}")
                        break
                    sleep_s = _backoff_seconds(
                        attempt,
                        cfg.siliconflow.initial_backoff_seconds,
                        cfg.siliconflow.max_backoff_seconds,
                    )
                    print(
                        f"Retry {attempt}/{cfg.siliconflow.max_retries} after {sleep_s}s: {last_error}"
                    )
                    time.sleep(sleep_s)
        elif backend == "diffusers":
            pipe, device, torch = client
            try:
                seed = cfg.diffusers.seed + int(prompt_id)
            except Exception:
                seed = cfg.diffusers.seed

            gen = torch.Generator(device=device).manual_seed(seed)
            try:
                out = pipe(
                    prompt_text,
                    num_inference_steps=cfg.diffusers.num_inference_steps,
                    guidance_scale=cfg.diffusers.guidance_scale,
                    width=cfg.diffusers.width,
                    height=cfg.diffusers.height,
                    generator=gen,
                )
                image = out.images[0]
                image.save(target_path)
                df.at[i, "image_path"] = target_path
                df.at[i, "generated_at"] = utc_now_iso()
                df.at[i, "error"] = ""
                generated += 1
            except Exception as e:
                last_error = str(e)
                df.at[i, "image_path"] = ""
                df.at[i, "generated_at"] = utc_now_iso()
                df.at[i, "error"] = last_error
                print(f"Failed prompt_id={prompt_id}: {last_error}")
        else:
            img_bytes = _dummy_image_bytes(
                width=cfg.dummy.width,
                height=cfg.dummy.height,
                font_size=cfg.dummy.font_size,
                text=prompt_text,
            )
            with open(target_path, "wb") as f:
                f.write(img_bytes)
            df.at[i, "image_path"] = target_path
            df.at[i, "generated_at"] = utc_now_iso()
            df.at[i, "error"] = ""
            generated += 1

        df.to_csv(cfg.paths.prompts_with_paths_csv, index=False, encoding="utf-8")

    df.to_csv(cfg.paths.prompts_with_paths_csv, index=False, encoding="utf-8")
    print(f"Done. Generated {generated}/{total} images.")
    print(f"Wrote -> {cfg.paths.prompts_with_paths_csv}")


if __name__ == "__main__":
    main()
