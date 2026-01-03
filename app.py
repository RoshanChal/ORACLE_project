import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from spillover_prediction import run_full_analysis

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Viral Spillover Risk Prediction",
    layout="wide"
)

st.title("Viral Spillover Risk Prediction")
st.markdown(
    """
    Paste a **viral entry protein sequence** below to estimate:
    - Spillover probability into humans
    - Confidence intervals
    - Plausible mutations and their effects
    """
)

# ===================== INPUT =====================
sequence = st.text_area(
    "Viral Entry Protein Sequence",
    height=250,
    placeholder="Paste amino acid sequence here (single-letter code)"
)

run = st.button("Analyze Spillover Risk")

# ===================== RUN ANALYSIS =====================
if run and sequence.strip():

    with st.spinner("Running ESM embedding + spillover model..."):
        base_prob, lo, hi, df = run_full_analysis(sequence.strip().upper())

    # ===================== RESULTS =====================
    st.success("Analysis complete!")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Baseline Spillover Probability",
        f"{base_prob:.3f}"
    )

    col2.metric(
        "Lower 95% CI",
        f"{lo:.3f}"
    )

    col3.metric(
        "Upper 95% CI",
        f"{hi:.3f}"
    )

    st.divider()

    # ===================== TABLE =====================
    st.subheader("Top Plausible Mutations")
    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    # ===================== BAR PLOT =====================
    st.subheader("Top Spillover-Increasing Mutations")

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    sns.barplot(
        data=df.head(15),
        x="Mutation",
        y="Delta",
        hue="Mutation",
        palette="coolwarm",
        legend=False,
        ax=ax1
    )
    ax1.set_ylabel("Δ Spillover Probability")
    ax1.set_xlabel("Mutation")
    plt.xticks(rotation=45)

    st.pyplot(fig1)

    # ===================== HEATMAP =====================
    st.subheader("Spillover Sensitivity Across Protein Positions")

    heat = df.pivot_table(
        index="Position",
        values="Delta",
        aggfunc="mean"
    )

    fig2, ax2 = plt.subplots(figsize=(4, 10))
    sns.heatmap(
        heat,
        cmap="coolwarm",
        center=0,
        ax=ax2
    )
    ax2.set_ylabel("Position")
    ax2.set_xlabel("Mean Δ Risk")

    st.pyplot(fig2)

else:
    st.info("Enter a sequence and click **Analyze Spillover Risk**.")