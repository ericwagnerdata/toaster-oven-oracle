"""
Streamlit UI. Run from repo root:
    streamlit run src/app.py
"""

import streamlit as st

from generate import answer

st.set_page_config(page_title="Toaster Oven Oracle", page_icon="🍕")
st.title("Toaster Oven Oracle")
st.caption("Ask questions about your Breville BOV900. Answers grounded in the manual.")

question = st.text_input(
    "Your question",
    placeholder="e.g. what's the convection bake setting for cookies?",
)

if question:
    with st.spinner("Thinking..."):
        result = answer(question)

    st.markdown("### Answer")
    st.write(result["answer"])

    with st.expander("Sources"):
        for i, c in enumerate(result["chunks"], start=1):
            st.markdown(f"**#{i} — page {c['page']}** (score {c['score']:.3f})")
            st.text(c["text"])
            st.divider()
