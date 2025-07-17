import os
import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scikit_posthocs as sp
import warnings
warnings.filterwarnings("ignore")

# Yükleme işlemleri
def load_results(base_dir, method_type):
    records = []
    for file in os.listdir(base_dir):
        if file.endswith("_acc.npy"):
            method = file.replace("_acc.npy", "")
            dataset, subset = method.split("_")[0], method.split("_")[1]
            accs = np.load(os.path.join(base_dir, file))
            for acc in accs:
                records.append({
                    "acc": acc,
                    "method": method,
                    "model": method.split("_")[2] if len(method.split("_")) > 2 else method.split("_")[0],
                    "dataset": dataset,
                    "subset": subset,
                    "type": method_type
                })
    return records

ml_path = "results_ml"
dl_path = "results_dl"
data = load_results(ml_path, "ML") + load_results(dl_path, "DL")
df = pd.DataFrame(data)
print("Toplam örnek sayısı:", len(df))

### 1. ML vs DL KARŞILAŞTIRMASI ###
print("\n\n--- [1] ML vs DL ---")
ml_acc = df[df["type"] == "ML"]["acc"]
dl_acc = df[df["type"] == "DL"]["acc"]
shapiro_ml = stats.shapiro(ml_acc)
shapiro_dl = stats.shapiro(dl_acc)
print(f"Shapiro ML: p={shapiro_ml.pvalue:.4f}, DL: p={shapiro_dl.pvalue:.4f}")
if shapiro_ml.pvalue > 0.05 and shapiro_dl.pvalue > 0.05:
    t_stat, p_val = stats.ttest_ind(ml_acc, dl_acc)
    print(f"Independent t-test: p={p_val:.4f}")
else:
    u_stat, p_val = stats.mannwhitneyu(ml_acc, dl_acc)
    print(f"Mann-Whitney U test: p={p_val:.4f}")

### 2. SET1–SET4 KARŞILAŞTIRMASI ###
print("\n\n--- [2] SET1–SET4 Comparison ---")
grouped = df.groupby("subset")
normals = {}
for name, group in grouped:
    pval = stats.shapiro(group["acc"]).pvalue
    normals[name] = pval
    print(f"{name} normality p={pval:.4f}")
if all(p > 0.05 for p in normals.values()):
    print("One-Way ANOVA:")
    anova = stats.f_oneway(*[g["acc"].values for name, g in grouped])
    print(f"ANOVA p={anova.pvalue:.4f}")
    if anova.pvalue < 0.05:
        tukey = pairwise_tukeyhsd(df["acc"], df["subset"])
        print(tukey.summary())
else:
    print("Kruskal–Wallis Test:")
    kw = stats.kruskal(*[g["acc"].values for name, g in grouped])
    print(f"Kruskal-Wallis p={kw.pvalue:.4f}")
    if kw.pvalue < 0.05:
        dunn = sp.posthoc_dunn(df, val_col="acc", group_col="subset", p_adjust="bonferroni")
        print("Dunn’s post-hoc test:\n", dunn)

### 3. TWO-WAY ANOVA: Model + Subset Etkisi ###
print("\n\n--- [3] Two-Way ANOVA: Model + Subset ---")
df['subset'] = df['subset'].astype(str)
df['model'] = df['model'].astype(str)
model = ols('acc ~ C(model) + C(subset) + C(model):C(subset)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)
