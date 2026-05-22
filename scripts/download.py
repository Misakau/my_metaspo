import json
import random
import re
from pathlib import Path

from datasets import load_dataset

OUT_ROOT = Path("datasets")
SEED = 42
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

random.seed(SEED)


def load_hf(path, name=None, **kwargs):
    try:
        if name is None:
            return load_dataset(path, trust_remote_code=True, **kwargs)
        return load_dataset(path, name, trust_remote_code=True, **kwargs)
    except TypeError:
        if name is None:
            return load_dataset(path, **kwargs)
        return load_dataset(path, name, **kwargs)


def norm(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()


def write_task(benchmark, task, train_rows, test_rows):
    out_dir = OUT_ROOT / benchmark
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{task}.json"

    data = {
        "train": train_rows,
        "test": test_rows,
    }

    with out_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[OK] {out_file} | train={len(train_rows)} test={len(test_rows)}")


def split_rows(rows, test_ratio=0.2):
    rows = list(rows)
    random.shuffle(rows)
    n_test = max(1, int(len(rows) * test_ratio))
    return rows[n_test:], rows[:n_test]


def get_split(ds, candidates):
    for name in candidates:
        if name in ds:
            return ds[name]
    return None


# -------------------------
# MedMCQA
# -------------------------

MEDMCQA_SUBJECTS = {
    "ob_gyn": ["obstetrics", "gynecology", "gynaecology", "o&g"],
    "medicine": ["medicine"],
    "pharmacology": ["pharmacology"],
    "pathology": ["pathology"],
    "dental": ["dental"],
    "anatomy": ["anatomy"],
    "surgery": ["surgery"],
    "pediatrics": ["pediatrics", "paediatrics"],
}


def convert_medmcqa_row(x):
    q = norm(x.get("question"))
    opts = [
        norm(x.get("opa")),
        norm(x.get("opb")),
        norm(x.get("opc")),
        norm(x.get("opd")),
    ]

    cop = x.get("cop")
    try:
        cop = int(cop)
    except Exception:
        return None

    # HF card says cop is 1,2,3,4.
    if cop not in [1, 2, 3, 4]:
        return None

    question = (
        f"Question: {q}\n"
        f"(A) {opts[0]}\n"
        f"(B) {opts[1]}\n"
        f"(C) {opts[2]}\n"
        f"(D) {opts[3]}"
    )
    answer = LETTERS[cop - 1]

    return {
        "question": question,
        "answer": answer,
    }


def download_medmcqa():
    print("\n=== Downloading MedMCQA ===")
    ds = load_hf("openlifescienceai/medmcqa")

    train_src = get_split(ds, ["train"])
    test_src = get_split(ds, ["validation", "dev", "test"])

    if train_src is None:
        raise RuntimeError("MedMCQA train split not found.")
    if test_src is None:
        test_src = train_src

    for task, keys in MEDMCQA_SUBJECTS.items():
        def match_subject(x):
            subj = norm(x.get("subject_name")).lower()
            return any(k in subj for k in keys)

        train_rows = []
        for x in train_src:
            if match_subject(x):
                y = convert_medmcqa_row(x)
                if y:
                    train_rows.append(y)

        test_rows = []
        for x in test_src:
            if match_subject(x):
                y = convert_medmcqa_row(x)
                if y:
                    test_rows.append(y)

        if not test_rows:
            train_rows, test_rows = split_rows(train_rows)

        write_task("medmcqa", task, train_rows, test_rows)


# -------------------------
# BIG-Bench
# -------------------------

BIGBENCH_MAP = {
    "logic_grid_puzzle": "logic_grid_puzzle",
    "tracking_shuffled_objects": "tracking_shuffled_objects",
    "logical_deduction": "logical_deduction",
    "temporal_sequences": "temporal_sequences",
    "object_counting": "object_counting",
    "reasoning_colored_objects": "reasoning_about_colored_objects",
    "epistemic": "epistemic_reasoning",
}

BIGBENCH_OPTION_TASKS = {
    "logical_deduction",
    "temporal_sequences",
    "tracking_shuffled_objects",
}


def first_existing(x, keys, default=None):
    for k in keys:
        if k in x and x[k] is not None:
            return x[k]
    return default


def as_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    try:
        return list(x)
    except Exception:
        return [x]


def convert_bigbench_row(x, task):
    inp = first_existing(x, ["inputs", "input", "question", "text"], "")
    inp = norm(inp)

    targets = as_list(first_existing(x, ["targets", "target", "answer"], []))
    choices = as_list(first_existing(x, ["multiple_choice_targets", "choices", "options"], []))
    scores = as_list(first_existing(x, ["multiple_choice_scores", "scores"], []))

    if choices and scores and len(choices) == len(scores):
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        correct_choice = norm(choices[best_idx])

        if task in BIGBENCH_OPTION_TASKS:
            option_text = "\n".join(
                f"({LETTERS[i]}) {norm(c)}"
                for i, c in enumerate(choices)
            )
            return {
                "question": f"{inp}\nOptions:\n{option_text}",
                "answer": LETTERS[best_idx],
            }

        return {
            "question": inp,
            "answer": correct_choice.lower(),
        }

    if targets:
        return {
            "question": inp,
            "answer": norm(targets[0]).lower(),
        }

    return None


def datasetdict_to_train_test(ds):
    train_split = get_split(ds, ["train", "default", "validation", "test"])
    test_split = get_split(ds, ["validation", "test", "default", "train"])

    if train_split is None:
        raise RuntimeError("No usable split found.")

    if test_split is None or test_split is train_split:
        rows = [dict(x) for x in train_split]
        return split_rows(rows)

    return list(train_split), list(test_split)


def download_bigbench():
    print("\n=== Downloading BIG-Bench ===")

    for task, hf_subset in BIGBENCH_MAP.items():
        ds = load_hf("google/bigbench", hf_subset)
        train_src, test_src = datasetdict_to_train_test(ds)

        train_rows = []
        for x in train_src:
            y = convert_bigbench_row(dict(x), task)
            if y:
                train_rows.append(y)

        test_rows = []
        for x in test_src:
            y = convert_bigbench_row(dict(x), task)
            if y:
                test_rows.append(y)

        if not test_rows:
            train_rows, test_rows = split_rows(train_rows)

        write_task("bigbench", task, train_rows, test_rows)


# -------------------------
# Grounding QA
# -------------------------

def qa_item(context, question, answers):
    answers = [norm(a) for a in as_list(answers) if norm(a)]
    if not question or not answers:
        return None

    context = norm(context)
    question = norm(question)

    if context:
        q = f"Context:\n{context}\n\nQuestion: {question}"
    else:
        q = f"Question: {question}"

    return {
        "question": q,
        "answers": answers,
    }


def convert_squad_row(x):
    answers = x.get("answers", {})
    if isinstance(answers, dict):
        answers = answers.get("text", [])
    return qa_item(x.get("context", ""), x.get("question", ""), answers)


def flatten_hotpot_context(ctx):
    if isinstance(ctx, dict):
        titles = ctx.get("title", [])
        sentences = ctx.get("sentences", [])
        parts = []
        for title, sents in zip(titles, sentences):
            parts.append(f"{title}: " + " ".join(map(str, sents)))
        return "\n".join(parts)
    return norm(ctx)


def convert_hotpot_row(x):
    return qa_item(flatten_hotpot_context(x.get("context", "")), x.get("question", ""), [x.get("answer", "")])


def convert_drop_row(x):
    answers = []
    answer = x.get("answer", None)

    if isinstance(answer, dict):
        spans = answer.get("spans") or []
        number = answer.get("number")
        date = answer.get("date")

        answers.extend(spans)
        if number:
            answers.append(number)
        if isinstance(date, dict):
            date_text = " ".join(norm(date.get(k)) for k in ["day", "month", "year"] if norm(date.get(k)))
            if date_text:
                answers.append(date_text)
    elif answer:
        answers.append(answer)

    return qa_item(x.get("passage", x.get("context", "")), x.get("question", ""), answers)


def convert_triviaqa_row(x):
    answer = x.get("answer", {})
    answers = []

    if isinstance(answer, dict):
        if answer.get("value"):
            answers.append(answer["value"])
        aliases = answer.get("aliases") or []
        answers.extend(aliases)
    else:
        answers.append(answer)

    context_parts = []

    for field in ["entity_pages", "search_results"]:
        obj = x.get(field)
        if isinstance(obj, dict):
            for k in ["wiki_context", "search_context", "doc_source", "title", "description"]:
                val = obj.get(k)
                if isinstance(val, list):
                    context_parts.extend(map(str, val[:5]))
                elif val:
                    context_parts.append(str(val))

    context = "\n".join(context_parts[:10])
    return qa_item(context, x.get("question", ""), answers)


def convert_nq_row(x):
    question = first_existing(x, ["question", "question_text"], "")

    # Prefer simplified SQuAD-like rows if available.
    context = first_existing(x, ["context", "document_text", "passage"], "")

    answers = []
    ans = x.get("answers")
    if isinstance(ans, dict):
        answers.extend(ans.get("text", []))
    elif ans:
        answers.extend(as_list(ans))

    # Google NQ raw format.
    annotations = x.get("annotations")
    if not answers and annotations:
        try:
            ann0 = annotations[0]
            short_answers = ann0.get("short_answers", [])
            for sa in short_answers:
                if isinstance(sa, dict) and sa.get("text"):
                    answers.append(sa["text"])
        except Exception:
            pass

    return qa_item(context, question, answers)


def convert_webq_row(x):
    question = first_existing(x, ["question", "utterance"], "")
    answers = first_existing(x, ["answers", "answer", "targetValue"], [])
    return qa_item("", question, answers)


def download_grounding():
    print("\n=== Downloading Grounding QA datasets ===")

    # SQuAD
    ds = load_hf("squad")
    train_src = get_split(ds, ["train"])
    test_src = get_split(ds, ["validation", "test"])
    write_task(
        "grounding",
        "squad",
        [y for x in train_src if (y := convert_squad_row(x))],
        [y for x in test_src if (y := convert_squad_row(x))],
    )

    # HotpotQA
    ds = load_hf("hotpotqa/hotpot_qa", "fullwiki")
    train_src = get_split(ds, ["train"])
    test_src = get_split(ds, ["validation", "test"])
    write_task(
        "grounding",
        "hotpot_qa",
        [y for x in train_src if (y := convert_hotpot_row(x))],
        [y for x in test_src if (y := convert_hotpot_row(x))],
    )

    # TriviaQA. rc.wikipedia includes context but is large.
    ds = load_hf("mandarjoshi/trivia_qa", "rc.wikipedia")
    train_src = get_split(ds, ["train"])
    test_src = get_split(ds, ["validation", "test"])
    write_task(
        "grounding",
        "trivia_qa",
        [y for x in train_src if (y := convert_triviaqa_row(x))],
        [y for x in test_src if (y := convert_triviaqa_row(x))],
    )

    # DROP
    try:
        ds = load_hf("ucinlp/drop")
    except Exception:
        ds = load_hf("drop")
    train_src = get_split(ds, ["train"])
    test_src = get_split(ds, ["validation", "test"])
    write_task(
        "grounding",
        "drop",
        [y for x in train_src if (y := convert_drop_row(x))],
        [y for x in test_src if (y := convert_drop_row(x))],
    )

    # Natural Questions. Try simplified first; fallback to raw Google NQ.
    try:
        ds = load_hf("LLukas22/nq-simplified")
    except Exception:
        ds = load_hf("google-research-datasets/natural_questions")
    train_src = get_split(ds, ["train"])
    test_src = get_split(ds, ["validation", "test"])
    if test_src is None:
        rows = [y for x in train_src if (y := convert_nq_row(x))]
        train_rows, test_rows = split_rows(rows)
    else:
        train_rows = [y for x in train_src if (y := convert_nq_row(x))]
        test_rows = [y for x in test_src if (y := convert_nq_row(x))]
    write_task("grounding", "natural_questions", train_rows, test_rows)

    # WebQuestions
    ds = load_hf("stanfordnlp/web_questions")
    train_src = get_split(ds, ["train"])
    test_src = get_split(ds, ["test", "validation"])
    if test_src is None:
        rows = [y for x in train_src if (y := convert_webq_row(x))]
        train_rows, test_rows = split_rows(rows)
    else:
        train_rows = [y for x in train_src if (y := convert_webq_row(x))]
        test_rows = [y for x in test_src if (y := convert_webq_row(x))]
    write_task("grounding", "web_qa", train_rows, test_rows)


# -------------------------
# Safety
# -------------------------

def safety_item(text, label):
    text = norm(text)
    label = norm(label).lower()
    if not text or label not in {"yes", "no"}:
        return None
    return {
        "question": text,
        "answer": label,
    }


def label_to_yes_no(v, positive_values={1, "1", True, "true", "yes", "hate", "hateful", "offensive", "sarcastic"}):
    if isinstance(v, str):
        vv = v.strip().lower()
        if vv in positive_values:
            return "yes"
        if vv in {"0", "false", "no", "not_hate", "non-hate", "normal", "not offensive", "not sarcastic"}:
            return "no"
        return "yes" if vv in positive_values else "no"
    return "yes" if v in positive_values else "no"


def convert_tweet_eval_row(x):
    label = label_to_yes_no(x.get("label"))
    return safety_item(x.get("text", ""), label)


def convert_hatecheck_row(x):
    label = first_existing(x, ["label_gold", "label", "target"], "")
    text = first_existing(x, ["test_case", "text", "sentence"], "")
    label_s = norm(label).lower()

    if label_s in {"hateful", "hate", "1", "yes"}:
        y = "yes"
    else:
        y = "no"

    return safety_item(text, y)


def convert_ethos_row(x):
    text = first_existing(x, ["text", "comment", "sentence"], "")
    label = first_existing(x, ["label", "hate_speech", "isHate"], 0)
    return safety_item(text, label_to_yes_no(label))


def convert_liar_row(x):
    statement = norm(x.get("statement", ""))
    subject = norm(x.get("subject", ""))
    speaker = norm(x.get("speaker", ""))
    context = norm(x.get("context", ""))

    question = (
        f"Statement: {statement}\n"
        f"Subject: {subject}\n"
        f"Speaker: {speaker}\n"
        f"Context: {context}"
    )

    label = x.get("label")
    label_name = norm(label).lower()

    # LIAR has 6-way labels. Treat false-ish labels as lie=yes.
    falseish = {"pants-fire", "false", "barely-true", "0", "1", "2"}
    trueish = {"half-true", "mostly-true", "true", "3", "4", "5"}

    if label_name in falseish:
        y = "yes"
    elif label_name in trueish:
        y = "no"
    else:
        try:
            y = "yes" if int(label) <= 2 else "no"
        except Exception:
            y = "no"

    return safety_item(question, y)


def convert_sarcasm_row(x):
    text = first_existing(x, ["text", "headline", "response", "Tweet", "tweet"], "")
    label = first_existing(x, ["label", "is_sarcastic", "sarcasm", "Label"], 0)

    if isinstance(label, str):
        y = "yes" if label.strip().lower() in {"1", "yes", "true", "sarcasm", "sarcastic"} else "no"
    else:
        y = "yes" if int(label) == 1 else "no"

    return safety_item(text, y)


def split_or_predefined(ds, converter):
    train_src = get_split(ds, ["train"])
    test_src = get_split(ds, ["test", "validation"])

    if train_src is not None and test_src is not None:
        return (
            [y for x in train_src if (y := converter(x))],
            [y for x in test_src if (y := converter(x))],
        )

    src = train_src or test_src or get_split(ds, ["validation"])
    rows = [y for x in src if (y := converter(x))]
    return split_rows(rows)


def download_safety():
    print("\n=== Downloading Safety datasets ===")

    # LIAR
    ds = load_hf("liar")
    train_rows, test_rows = split_or_predefined(ds, convert_liar_row)
    write_task("safety", "liar", train_rows, test_rows)

    # HateCheck
    ds = load_hf("Paul/hatecheck")
    rows = []
    for split_name in ds:
        rows.extend([y for x in ds[split_name] if (y := convert_hatecheck_row(x))])
    train_rows, test_rows = split_rows(rows)
    write_task("safety", "hatecheck", train_rows, test_rows)

    # TweetEval offensive
    try:
        ds = load_hf("tweet_eval", "offensive")
    except Exception:
        ds = load_hf("cardiffnlp/tweet_eval", "offensive")
    train_rows, test_rows = split_or_predefined(ds, convert_tweet_eval_row)
    write_task("safety", "tweet_eval", train_rows, test_rows)

    # Sarcasm. Prefer FIGLANG sarcasm; fallback to headlines dataset.
    try:
        ds = load_hf("tasksource/figlang2020-sarcasm")
        rows = []
        for split_name in ds:
            rows.extend([y for x in ds[split_name] if (y := convert_sarcasm_row(x))])
        train_rows, test_rows = split_rows(rows)
    except Exception:
        ds = load_hf("raquiba/Sarcasm_News_Headline")
        train_rows, test_rows = split_or_predefined(ds, convert_sarcasm_row)
    write_task("safety", "sarcasm", train_rows, test_rows)

    # ETHOS binary
    try:
        ds = load_hf("SetFit/ethos_binary")
    except Exception:
        ds = load_hf("iamollas/ethos")
    train_rows, test_rows = split_or_predefined(ds, convert_ethos_row)
    write_task("safety", "ethos", train_rows, test_rows)

    # Anthropic harmless. Project spelling is antropic_harmless.
    ds = load_hf("Anthropic/hh-rlhf")

    rows = []
    for split_name in ds:
        if "harmless" not in split_name.lower() and split_name not in {"train", "test"}:
            continue

        for x in ds[split_name]:
            chosen = norm(x.get("chosen", ""))
            rejected = norm(x.get("rejected", ""))

            if chosen:
                rows.append(safety_item(chosen, "no"))
            if rejected:
                rows.append(safety_item(rejected, "yes"))

    rows = [x for x in rows if x]
    train_rows, test_rows = split_rows(rows)
    write_task("safety", "antropic_harmless", train_rows, test_rows)


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # download_medmcqa()
    # download_bigbench()
    # download_grounding()
    download_safety()

    print("\nDONE. Generated files:")
    for p in sorted(OUT_ROOT.glob("*/*.json")):
        print(p)


if __name__ == "__main__":
    main()