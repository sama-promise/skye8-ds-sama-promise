import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import statsmodels.formula.api as smf

pd.set_option("display.max_columns", None)


def normalise_boolean(series: pd.Series) -> pd.Series:
    mapping = {"true": True, "yes": True, "1": True,
               "false": False, "no": False, "0": False}
    text = series.astype("string").str.strip().str.lower()
    return text.map(mapping).astype("boolean")


def normalise_number(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(text, errors="coerce")


# --- Load ---
df1 = pd.read_csv("./data/raw/listings.csv")
df2 = pd.read_csv("./data/raw/neighbourhoods.csv")

# --- Clean booleans and numbers ---
df1["has_generator"] = normalise_boolean(df1["has_generator"])
df1["has_borehole"] = normalise_boolean(df1["has_borehole"])
df1["tiled"] = normalise_boolean(df1["tiled"])
df1["gated_security"] = normalise_boolean(df1["gated_security"])
df1["furnished"] = normalise_boolean(df1["furnished"])
df2["flood_prone"] = normalise_boolean(df2["flood_prone"])

df1["area_sqm"] = normalise_number(df1["area_sqm"])
df1["monthly_rent_xaf"] = normalise_number(df1["monthly_rent_xaf"])

print(df1.dtypes)
print(df2.dtypes)

# --- Missing rent investigation ---
df1["rent_missing"] = df1["monthly_rent_xaf"].isna()
missing_rate_by_source = df1.groupby("listing_source")["rent_missing"].mean()
print(missing_rate_by_source)
print(df1["listing_source"].value_counts())
print(df1.groupby("listing_source")["rent_missing"].sum())

known = df1[df1["monthly_rent_xaf"].notna()].copy()
print(len(df1))
print(len(known))

# --- Rent distribution ---
mean_rent = known["monthly_rent_xaf"].mean()
median_rent = known["monthly_rent_xaf"].median()
q1 = known["monthly_rent_xaf"].quantile(0.25)
q3 = known["monthly_rent_xaf"].quantile(0.75)
skewness = known["monthly_rent_xaf"].skew()
print("mean:", mean_rent)
print("median:", median_rent)
print("Q1:", q1)
print("Q3:", q3)
print("skewness:", skewness)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(known["monthly_rent_xaf"], bins=50)
axes[0].set_title("Monthly rent (raw scale)")
axes[0].set_xlabel("XAF")
axes[1].hist(known["monthly_rent_xaf"], bins=50)
axes[1].set_xscale("log")
axes[1].set_title("Monthly rent (log scale)")
axes[1].set_xlabel("XAF (log scale)")
plt.tight_layout()
plt.savefig("rent_distribution.png")
print("saved rent_distribution.png")

# --- Duplicates ---
duplicate_count = df1.duplicated().sum()
print("duplicate rows:", duplicate_count)
df1 = df1.drop_duplicates()
known = df1[df1["monthly_rent_xaf"].notna()].copy()

# --- Dates ---
sample_dates = df1["listed_on"].drop_duplicates().sample(15, random_state=1)
print(sample_dates.tolist())
df1["listed_on"] = pd.to_datetime(df1["listed_on"], format="mixed", dayfirst=True, errors="coerce")
print("unparseable dates:", df1["listed_on"].isna().sum())

# --- Collinearity check ---
numeric_cols = ["bedrooms", "bathrooms", "area_sqm", "building_age_years", "floor"]
print(df1[numeric_cols].corr())

# --- Predictor-vs-rent plots ---
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
numeric_predictors = ["bedrooms", "bathrooms", "area_sqm", "building_age_years", "floor"]
for ax, col in zip(axes.ravel(), numeric_predictors):
    ax.scatter(known[col], known["monthly_rent_xaf"], alpha=0.2, s=8)
    ax.set_xlabel(col)
    ax.set_ylabel("monthly rent (XAF)")
plt.tight_layout()
plt.savefig("predictors_vs_rent.png")
print("saved predictors_vs_rent.png")

# --- Extreme listings ---
rent_z = (known["monthly_rent_xaf"] - known["monthly_rent_xaf"].mean()) / known["monthly_rent_xaf"].std()
area_z = (known["area_sqm"] - known["area_sqm"].mean()) / known["area_sqm"].std()
extremes = known[(rent_z.abs() >= 3) | (area_z.abs() >= 3)]
print("extreme listings:", len(extremes))
extremes[["listing_id", "city", "area_sqm", "monthly_rent_xaf"]].to_csv("extreme_listings.csv", index=False)
print("saved extreme_listings.csv")

# =========================
# STAGE B
# =========================

neighbourhood_summary = known.groupby("neighbourhood")["monthly_rent_xaf"].agg(
    count="count", median="median",
    q1=lambda x: x.quantile(0.25), q3=lambda x: x.quantile(0.75)
)
neighbourhood_summary["iqr"] = neighbourhood_summary["q3"] - neighbourhood_summary["q1"]
print(neighbourhood_summary)
neighbourhood_summary.to_csv("rent_by_neighbourhood.csv")

property_summary = known.groupby("property_type")["monthly_rent_xaf"].agg(
    count="count", median="median",
    q1=lambda x: x.quantile(0.25), q3=lambda x: x.quantile(0.75)
)
property_summary["iqr"] = property_summary["q3"] - property_summary["q1"]
print(property_summary)
property_summary.to_csv("rent_by_property_type.csv")


def bootstrap_group_diff(values_a, values_b, n_boot=5000, seed=42):
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        a_sample = rng.choice(values_a, size=len(values_a), replace=True)
        b_sample = rng.choice(values_b, size=len(values_b), replace=True)
        diffs[i] = np.median(a_sample) - np.median(b_sample)
    return diffs


douala = known.loc[known.city == "Douala", "monthly_rent_xaf"].values
yaounde = known.loc[known.city == "Yaounde", "monthly_rent_xaf"].values
diffs = bootstrap_group_diff(douala, yaounde)
observed_diff = np.median(douala) - np.median(yaounde)
ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
print("observed median diff (Douala - Yaounde):", observed_diff)
print("95% CI:", ci_low, ci_high)

with_gen = known.loc[known.has_generator == True, "monthly_rent_xaf"].dropna().values
without_gen = known.loc[known.has_generator == False, "monthly_rent_xaf"].dropna().values
gen_diffs = bootstrap_group_diff(with_gen, without_gen)
gen_observed = np.median(with_gen) - np.median(without_gen)
gen_ci_low, gen_ci_high = np.percentile(gen_diffs, [2.5, 97.5])
print("generator median rent diff:", gen_observed)
print("95% CI:", gen_ci_low, gen_ci_high)

crosstab = pd.crosstab(known["neighbourhood"], known["has_generator"])
chi2, p_value, dof, expected = chi2_contingency(crosstab)
print("chi-square p-value for generator vs neighbourhood:", p_value)

n_tests = 3
adjusted_ci = 100 * (1 - 0.05 / n_tests)
print("Bonferroni-adjusted CI level:", adjusted_ci)
city_ci_adj = np.percentile(diffs, [(0.05 / n_tests / 2) * 100, 100 - (0.05 / n_tests / 2) * 100])
gen_ci_adj = np.percentile(gen_diffs, [(0.05 / n_tests / 2) * 100, 100 - (0.05 / n_tests / 2) * 100])
print("city diff, Bonferroni-adjusted CI:", city_ci_adj)
print("generator diff, Bonferroni-adjusted CI:", gen_ci_adj)
adjusted_alpha = 0.05 / n_tests
print("generator-neighbourhood p-value still significant after correction:", p_value < adjusted_alpha)

# =========================
# STAGE C
# =========================

bool_cols = ["has_generator", "has_borehole", "tiled", "gated_security", "furnished"]
print(known[bool_cols].isna().sum())
known[bool_cols] = known[bool_cols].astype(bool)

model_formula = (
    "monthly_rent_xaf ~ bedrooms + bathrooms + area_sqm + building_age_years + floor "
    "+ has_generator + has_borehole + tiled + gated_security + furnished "
    "+ C(property_type) + C(neighbourhood)"
)
model = smf.ols(model_formula, data=known).fit()
print(model.summary())

fitted = model.fittedvalues
residuals = model.resid

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].scatter(fitted, residuals, alpha=0.2, s=8)
axes[0].axhline(0, color="red", linewidth=1)
axes[0].set_xlabel("fitted values")
axes[0].set_ylabel("residuals")
axes[0].set_title("Residuals vs fitted")

import scipy.stats as stats
stats.probplot(residuals, dist="norm", plot=axes[1])
axes[1].set_title("Normal Q-Q plot of residuals")

axes[2].scatter(known["area_sqm"], residuals, alpha=0.2, s=8)
axes[2].axhline(0, color="red", linewidth=1)
axes[2].set_xlabel("area_sqm")
axes[2].set_ylabel("residuals")
axes[2].set_title("Residuals vs area_sqm")

plt.tight_layout()
plt.savefig("residuals_raw_model.png")
print("saved residuals_raw_model.png")

known["log_rent"] = np.log(known["monthly_rent_xaf"])

log_formula = (
    "log_rent ~ bedrooms + bathrooms + area_sqm + building_age_years + floor "
    "+ has_generator + has_borehole + tiled + gated_security + furnished "
    "+ C(property_type) + C(neighbourhood)"
)
log_model = smf.ols(log_formula, data=known).fit()
print(log_model.summary())

log_fitted = log_model.fittedvalues
log_residuals = log_model.resid

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(log_fitted, log_residuals, alpha=0.2, s=8)
axes[0].axhline(0, color="red", linewidth=1)
axes[0].set_xlabel("fitted values (log rent)")
axes[0].set_ylabel("residuals")
axes[0].set_title("Residuals vs fitted (log model)")

stats.probplot(log_residuals, dist="norm", plot=axes[1])
axes[1].set_title("Normal Q-Q plot (log model)")

plt.tight_layout()
plt.savefig("residuals_log_model.png")
print("saved residuals_log_model.png")