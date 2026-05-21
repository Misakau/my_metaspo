"""Sample 3-5 entries from every dataset this project touches and dump them to disk.

Local datasets (./datasets/amazon/*.json) are read directly. Remote datasets are
fetched from Hugging Face in streaming mode (no full download). HF dataset paths
below are best-guess mappings from the project's task names — a failure for any
single task is logged and the script keeps going, so you can fix the mapping and
rerun without losing the rest.

Output: ./dataset_samples/<benchmark>/<task_name>.json
"""

import json
import os
import sys
import traceback
from itertools import islice
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_DATA_DIR = REPO_ROOT / "datasets"
OUTPUT_DIR = REPO_ROOT / "dataset_samples"
SAMPLES_PER_TASK = 5


# (benchmark, task_name, hf_path, config_or_subset, split, filter_fn)
# filter_fn: callable(example) -> bool, or None
# Some HF paths require trust_remote_code=True; we set it globally.
HF_DATASETS = [
    # ---- BIG-Bench (tasksource/bigbench mirrors the original) ----
    ("bigbench", "logic_grid_puzzle",          "tasksource/bigbench", "logic_grid_puzzle",          "train", None),
    ("bigbench", "logical_deduction",          "tasksource/bigbench", "logical_deduction_seven_objects", "train", None),
    ("bigbench", "temporal_sequences",         "tasksource/bigbench", "temporal_sequences",         "train", None),
    ("bigbench", "tracking_shuffled_objects",  "tasksource/bigbench", "tracking_shuffled_objects_seven_objects", "train", None),
    ("bigbench", "object_counting",            "tasksource/bigbench", "object_counting",            "train", None),
    ("bigbench", "reasoning_colored_objects",  "tasksource/bigbench", "reasoning_about_colored_objects", "train", None),
    ("bigbench", "epistemic",                  "tasksource/bigbench", "epistemic_reasoning",        "train", None),
    ("bigbench", "navigate",                   "tasksource/bigbench", "navigate",                   "train", None),

    # ---- Safety ----
    ("safety", "ethos",              "iamollas/ethos",            "binary",      "train", None),
    ("safety", "liar",               "liar",                      None,          "train", None),
    ("safety", "hatecheck",          "Paul/hatecheck",            None,          "test",  None),
    ("safety", "sarcasm",            "raquiba/Sarcasm_News_Headline", None,      "train", None),
    ("safety", "tweet_eval",         "tweet_eval",                "offensive",   "train", None),
    ("safety", "antropic_harmless",  "Anthropic/hh-rlhf",         "harmless-base", "train", None),

    # ---- Grounding / QA ----
    ("grounding", "squad",              "squad",              None, "train",      None),
    ("grounding", "hotpot_qa",          "hotpot_qa",          "distractor", "train", None),
    ("grounding", "trivia_qa",          "trivia_qa",          "rc.nocontext",  "train", None),
    ("grounding", "drop",               "drop",               None, "train",      None),
    ("grounding", "natural_questions",  "nq_open",            None, "train",      None),
    ("grounding", "web_qa",             "web_questions",      None, "train",      None),

    # ---- MedMCQA: one source dataset, filter by subject_name for each task ----
    ("medmcqa", "anatomy",      "openlifescienceai/medmcqa", None, "train", lambda ex: (ex.get("subject_name") or "").lower() == "anatomy"),
    ("medmcqa", "surgery",      "openlifescienceai/medmcqa", None, "train", lambda ex: (ex.get("subject_name") or "").lower() == "surgery"),
    ("medmcqa", "ob_gyn",       "openlifescienceai/medmcqa", None, "train", lambda ex: "gynaecology" in (ex.get("subject_name") or "").lower() or "obstetrics" in (ex.get("subject_name") or "").lower()),
    ("medmcqa", "medicine",     "openlifescienceai/medmcqa", None, "train", lambda ex: (ex.get("subject_name") or "").lower() == "medicine"),
    ("medmcqa", "pharmacology", "openlifescienceai/medmcqa", None, "train", lambda ex: (ex.get("subject_name") or "").lower() == "pharmacology"),
    ("medmcqa", "dental",       "openlifescienceai/medmcqa", None, "train", lambda ex: (ex.get("subject_name") or "").lower() == "dental"),
    ("medmcqa", "pediatrics",   "openlifescienceai/medmcqa", None, "train", lambda ex: (ex.get("subject_name") or "").lower() == "pediatrics"),
    ("medmcqa", "pathology",    "openlifescienceai/medmcqa", None, "train", lambda ex: (ex.get("subject_name") or "").lower() == "pathology"),
]


def _to_jsonable(obj):
    """Convert non-JSON-serializable values (numpy, bytes, etc.) to plain types."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (bytes, bytearray)):
        try:
            return obj.decode("utf-8", errors="replace")
        except Exception:
            return repr(obj)
    if hasattr(obj, "tolist"):  # numpy array / scalar
        return obj.tolist()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _save(out_path: Path, samples, source: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"source": source, "num_samples": len(samples), "samples": _to_jsonable(samples)}
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def sample_local_amazon():
    benchmark = "amazon"
    src_dir = LOCAL_DATA_DIR / benchmark
    if not src_dir.exists():
        print(f"[skip] local dir not found: {src_dir}")
        return

    for json_file in sorted(src_dir.glob("*.json")):
        task = json_file.stem
        out = OUTPUT_DIR / benchmark / f"{task}.json"
        try:
            with json_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            train = data.get("train", []) if isinstance(data, dict) else []
            samples = train[:SAMPLES_PER_TASK]
            _save(out, samples, source=f"local:{json_file.relative_to(REPO_ROOT)}")
            print(f"[ok ] amazon/{task}: {len(samples)} samples -> {out.relative_to(REPO_ROOT)}")
        except Exception as e:
            print(f"[err] amazon/{task}: {e}")


def sample_hf():
    try:
        from datasets import load_dataset
    except ImportError:
        print("[fatal] `datasets` not installed; run `pip install datasets`")
        return

    for benchmark, task, hf_path, config, split, filter_fn in HF_DATASETS:
        out = OUTPUT_DIR / benchmark / f"{task}.json"
        label = f"{benchmark}/{task}"
        source = f"hf:{hf_path}" + (f":{config}" if config else "") + f"[{split}]"
        try:
            ds = load_dataset(
                hf_path,
                name=config,
                split=split,
                streaming=True,
                trust_remote_code=True,
            )
            stream = ds if filter_fn is None else (ex for ex in ds if filter_fn(ex))
            samples = list(islice(stream, SAMPLES_PER_TASK))
            if not samples:
                raise RuntimeError("empty after filtering — check filter_fn or dataset path")
            _save(out, samples, source=source)
            print(f"[ok ] {label}: {len(samples)} samples <- {source}")
        except Exception as e:
            print(f"[err] {label}: {source}: {e.__class__.__name__}: {e}")
            # Optionally write a stub explaining the failure so it's visible on disk.
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", encoding="utf-8") as f:
                json.dump({"source": source, "error": f"{e.__class__.__name__}: {e}", "samples": []}, f, indent=2)


def main():
    print(f"writing samples to {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sample_local_amazon()
    sample_hf()
    print("done.")


if __name__ == "__main__":
    main()
