import numpy as np
import torch
import esm
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ===================== CONFIG =====================
MODEL_BUNDLE = "spillover_classifier.joblib"
ESM_MODEL_NAME = "esm2_t30_150M_UR50D"
DEVICE = "cpu"

TOP_K_MUT = 3
MIN_MUT_PROB = 0.05
MAX_MUTATIONS = 40
ENTROPY_PERCENTILE = 75
N_BOOTSTRAP = 300
# =================================================


# ===================== LOAD MODELS =====================
print("Loading spillover model...")
bundle = joblib.load(MODEL_BUNDLE)
clf = bundle["model"]
nonhuman_mean = bundle["nonhuman_mean"]

print("Loading ESM model...")
esm_model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
esm_model = esm_model.to(DEVICE)
esm_model.eval()

batch_converter = alphabet.get_batch_converter()
mask_idx = alphabet.mask_idx
REP_LAYER = esm_model.num_layers
# =====================================================


# ===================== EMBEDDING =====================
def embed_sequence(seq):
    seq = seq.replace("\n", "").strip().upper()
    batch = [("protein", seq)]
    _, _, tokens = batch_converter(batch)
    tokens = tokens.to(DEVICE)

    with torch.no_grad():
        out = esm_model(tokens, repr_layers=[REP_LAYER])

    reps = out["representations"][REP_LAYER][0, 1:-1]
    return reps.mean(dim=0).cpu().numpy()
# ====================================================


# ===================== CONFIDENCE INTERVAL =====================
def predict_with_ci(embedding, n_samples=200, noise_scale=0.02):
    delta = embedding - nonhuman_mean
    base_feat = np.concatenate([delta, delta])

    probs = []

    for _ in range(n_samples):
        noise = np.random.normal(0, noise_scale, size=base_feat.shape)
        feat = (base_feat + noise).reshape(1, -1)
        p = clf.predict_proba(feat)[0, 1]
        probs.append(p)

    probs = np.array(probs)

    mean = probs.mean()
    lo, hi = np.percentile(probs, [2.5, 97.5])

    return mean, lo, hi
# =============================================================


# ===================== POSITION ENTROPY =====================
def compute_position_entropy(sequence):
    data = [("seq", sequence)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(DEVICE)

    entropies = []

    with torch.no_grad():
        for pos in range(len(sequence)):
            masked = tokens.clone()
            masked[0, pos + 1] = mask_idx

            logits = esm_model(masked)["logits"][0, pos + 1]
            probs = torch.softmax(logits, dim=0)

            aa_probs = []
            for i, p in enumerate(probs):
                tok = alphabet.get_tok(i)
                if len(tok) == 1 and tok.isalpha():
                    aa_probs.append(p.item())

            aa_probs = np.array(aa_probs)
            aa_probs /= aa_probs.sum()
            entropy = -np.sum(aa_probs * np.log(aa_probs + 1e-9))
            entropies.append(entropy)

    return np.array(entropies)
# ============================================================


# ===================== MUTATION GENERATION =====================
def generate_mutations(sequence):
    entropy = compute_position_entropy(sequence)
    threshold = np.percentile(entropy, ENTROPY_PERCENTILE)
    allowed_positions = np.where(entropy >= threshold)[0]

    mutations = []

    data = [("seq", sequence)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(DEVICE)

    with torch.no_grad():
        for pos in allowed_positions:
            masked = tokens.clone()
            masked[0, pos + 1] = mask_idx

            logits = esm_model(masked)["logits"][0, pos + 1]
            probs = torch.softmax(logits, dim=0)

            top_p, top_i = torch.topk(probs, TOP_K_MUT)

            for p, i in zip(top_p, top_i):
                aa = alphabet.get_tok(i)
                if aa == sequence[pos]:
                    continue
                if p.item() < MIN_MUT_PROB:
                    continue
                if not aa.isalpha():
                    continue

                mutated = sequence[:pos] + aa + sequence[pos + 1:]
                mutations.append({
                    "Position": pos + 1,
                    "Mutation": f"{sequence[pos]}{pos+1}{aa}",
                    "ESM_prob": p.item(),
                    "Sequence": mutated,
                    "Entropy": entropy[pos]
                })

            if len(mutations) >= MAX_MUTATIONS:
                break

    return mutations
# =============================================================


# ===================== FULL PIPELINE =====================
def run_full_analysis(sequence):
    print("\nRunning spillover analysis...")

    base_emb = embed_sequence(sequence)
    base_prob, base_lo, base_hi = predict_with_ci(base_emb)

    print("\nBASELINE RISK")
    print(f"Probability: {base_prob:.3f}")
    print(f"95% CI: [{base_lo:.3f}, {base_hi:.3f}]")

    muts = generate_mutations(sequence)
    rows = []

    for m in muts:
        emb = embed_sequence(m["Sequence"])
        prob, lo, hi = predict_with_ci(emb)

        rows.append({
            "Position": m["Position"],
            "Mutation": m["Mutation"],
            "ESM_prob": m["ESM_prob"],
            "Entropy": m["Entropy"],
            "Spillover_prob": prob,
            "Delta": prob - base_prob,
            "CI_low": lo,
            "CI_high": hi
        })

    df = pd.DataFrame(rows).sort_values("Delta", ascending=False)
    return base_prob, base_lo, base_hi, df
# =======================================================


# ===================== VISUALIZATIONS =====================
def make_plots(df):
    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=df.head(15),
        x="Mutation",
        y="Delta",
        hue="Mutation",
        palette="coolwarm",
        legend=False
    )
    plt.xticks(rotation=45)
    plt.title("Top Spillover-Increasing Mutations")
    plt.tight_layout()
    plt.show()

    heat = df.pivot_table(
        index="Position",
        values="Delta",
        aggfunc="mean"
    )

    plt.figure(figsize=(4, 10))
    sns.heatmap(heat, cmap="coolwarm", center=0)
    plt.title("Delta Spillover Risk by Position")
    plt.tight_layout()
    plt.show()
# =========================================================


# ===================== RUN =====================
if __name__ == "__main__":
    seq = input("\nPaste viral entry protein sequence:\n").strip().upper()

    base_prob, lo, hi, df = run_full_analysis(seq)

    print("\nTOP MUTATIONS")
    print(df.head(10))

    df.to_csv("mutation_spillover_results.csv", index=False)
    print("\nSaved: mutation_spillover_results.csv")

    make_plots(df)