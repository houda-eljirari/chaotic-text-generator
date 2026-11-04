"""Lab 03 - Chaotic Text Generator (Streamlit UI). Run:  streamlit run app.py"""
import random

import streamlit as st

from src.pipeline import HMM, NGramGenerator, load_tagged, split_by_document

st.set_page_config(page_title="Chaotic Text Generator")
st.title("Chaotic Text Generator")
st.caption("Left: POS-HMM (hidden tags -> words).  Right: trigram model.  Same temperature, same seed.")


@st.cache_resource
def get_models():
    docs = load_tagged()
    train, _ = split_by_document(docs)
    return HMM(train), HMM(train, split_states=True), NGramGenerator([w for w, _ in train])


hmm_plain, hmm_split, ng = get_models()

c1, c2, c3 = st.columns(3)
temperature = c1.slider("Temperature (chaos)", 0.3, 3.0, 1.0, 0.1)
seed = c2.number_input("Random seed", 0, 10_000, 42)
prefix_text = c3.text_input("Trigram prompt (optional)", "the")
split = st.checkbox("HMM with role-split states (S / O / P) - better sentence structure", value=False)
hmm = hmm_split if split else hmm_plain

prefix = prefix_text.lower().split()
unknown = [w for w in prefix if w not in ng.vocab]
if unknown:
    st.error(f"Unknown word(s) in the prompt: {', '.join(unknown)}. Use words from the training vocabulary, "
             f"e.g. the, cats, farmers, garden.")
else:
    if st.button("Generate two samples", type="primary"):
        rng = random.Random(int(seed))
        left, right = st.columns(2)
        for i in range(2):
            words, states = hmm.sample_with_states(rng, temperature)
            left.markdown(f"**{' '.join(words)}**")
            left.caption(" → ".join(states))
        for i in range(2):
            words = ng.sample_sentence(rng, temperature, prefix=prefix)
            right.markdown(f"**{' '.join(words)}**")
        st.caption("A POS-HMM only knows grammar patterns, not meaning: expect fluent nonsense such as "
                   "'the apples sleep in the school'. That is the point of the lab.")
