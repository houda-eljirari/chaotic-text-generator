"""Lab 03 - Chaotic Text Generator: test bench for text generators. Run:  streamlit run app.py"""
import random
import re

import pandas as pd
import streamlit as st

from src.pipeline import (HMM, NGramGenerator, coarse, generation_metrics,
                          load_tagged, split_by_document)

st.set_page_config(page_title="Chaotic Text Generator", layout="wide")
st.title("Chaotic Text Generator - test bench")

NOTICE = ("The corpus is synthetic and the grammar is tiny. Metrics such as valid tag pattern are "
          "defined relative to this corpus and do not measure the quality of real English.")
st.warning(NOTICE)

MODELS = ["HMM (plain states)", "HMM (role-split states)", "Trigram"]
METRIC_LABELS = {"valid_tag_pattern": "Valid tag pattern",
                 "plausible_subject_verb": "Plausible subject-verb",
                 "novel_sentences": "Novel sentences",
                 "distinct_2": "Distinct-2"}


@st.cache_resource
def get_models():
    docs = load_tagged()
    train, _ = split_by_document(docs)
    plain, split = HMM(train), HMM(train, split_states=True)
    ng = NGramGenerator([w for w, _ in train])
    return train, plain, split, ng


train, hmm_plain, hmm_split, ngram = get_models()
train_text = {" ".join(w) for w, _ in train}
train_patterns = {" ".join(t) for _, t in train}


def generate(model, n, temperature, seed):
    """Return a list of (words, coarse_tags, hidden_states) for n sentences."""
    rng = random.Random(int(seed))
    out = []
    for _ in range(n):
        if model == MODELS[2]:
            words = ngram.sample_sentence(rng, temperature)
            tags = hmm_plain.viterbi(words)          # tag the trigram output with Viterbi
            states = tags
        else:
            hmm = hmm_split if model == MODELS[1] else hmm_plain
            words, states = hmm.sample_with_states(rng, temperature)
            tags = coarse(states)
        out.append((words, tags, states))
    return out


def metrics_of(samples):
    return generation_metrics([(w, t) for w, t, _ in samples], train, hmm_plain)


# ------------------------------------------------------------------ 1. control panel
st.header("1. Control panel")
c1, c2, c3, c4 = st.columns(4)
model = c1.selectbox("Model", MODELS)
temperature = c2.slider("Temperature", 0.3, 3.0, 1.0, 0.1)
seed = c3.number_input("Random seed", 0, 10_000, 42)
n_sent = c4.slider("Number of sentences", 10, 500, 100, 10)

# ------------------------------------------------------------------ 2. generate
if st.button("Generate", type="primary"):
    samples = generate(model, n_sent, temperature, seed)
    rows = []
    for i, (w, t, s) in enumerate(samples, 1):
        pattern = " ".join(t)
        rows.append({"#": i, "sentence": " ".join(w), "hidden tags": " ".join(s),
                     "valid pattern": "✅" if pattern in train_patterns else "❌",
                     "copied from train": "yes" if " ".join(w) in train_text else "no"})
    st.session_state["result"] = {"df": pd.DataFrame(rows), "metrics": metrics_of(samples),
                                  "settings": (model, temperature, int(seed), n_sent)}

res = st.session_state.get("result")
if res:
    m_name, m_temp, m_seed, m_n = res["settings"]
    st.subheader(f"Generated sentences - {m_name}, T={m_temp}, seed={m_seed}, n={m_n}")
    st.dataframe(res["df"], hide_index=True)

    # -------------------------------------------------------------- 3. metrics
    st.header("2. Metrics of the generated batch")
    cols = st.columns(4)
    for col, (key, label) in zip(cols, METRIC_LABELS.items()):
        v = res["metrics"][key]
        col.metric(label, f"{v:.3f}" if key == "distinct_2" else f"{v:.1%}")

    # -------------------------------------------------------------- 5. CSV
    csv = res["df"].assign(model=m_name, temperature=m_temp, seed=m_seed).to_csv(index=False)
    st.download_button("Download CSV", csv, file_name="generated_sentences.csv", mime="text/csv")
else:
    st.info("Choose the settings and click Generate.")

# ------------------------------------------------------------------ optional: comparison
st.header("3. Compare the three models (optional)")
oc1, oc2 = st.columns(2)
if oc1.button("Compare the 3 models at the current settings"):
    comp = {}
    for mod in MODELS:
        comp[mod] = metrics_of(generate(mod, n_sent, temperature, seed))
    df = pd.DataFrame(comp).T.rename(columns=METRIC_LABELS)
    st.dataframe(df.style.format("{:.3f}"))
if oc2.button("Plot valid tag pattern vs temperature"):
    temps = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    curve = {mod: [metrics_of(generate(mod, n_sent, t, seed))["valid_tag_pattern"] for t in temps]
             for mod in MODELS}
    st.line_chart(pd.DataFrame(curve, index=temps).rename_axis("temperature"))

# ------------------------------------------------------------------ 4. tagger
st.header("4. Viterbi tagger")
text = st.text_input("Type a sentence", "the old farmers watch the birds .")
tokens = re.findall(r"\w+|[^\w\s]", text.lower())[:40]
if not tokens:
    st.info("Type a sentence to see its tags.")
else:
    unknown = [w for w in tokens if w not in hmm_plain.vocab]
    if unknown:
        st.warning(f"Unknown word(s), not in the training vocabulary: {', '.join(sorted(set(unknown)))}. "
                   "The tag below is a guess and may be unreliable.")
    try:
        tags = hmm_plain.viterbi(tokens)
        shown = [f"{w} (?)" if w in unknown else w for w in tokens]
        table = pd.DataFrame([shown, tags], index=["word", "Viterbi tag"],
                             columns=range(1, len(tokens) + 1))
        st.dataframe(table)
    except Exception as exc:  # never crash the app on user input
        st.error(f"Could not tag this sentence: {exc}")

st.caption("A generated sentence is fluent but has no meaning or intent. Do not present generated text "
           "as written by a person.")