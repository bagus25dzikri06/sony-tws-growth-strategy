
import numpy as np
import pandas as pd
import statsmodels.api as sm

# ==========================================
# 1. GENERATE 500 HYPOTHETICAL RESPONDENTS
# ==========================================

np.random.seed(42)
n = 500

age = np.clip(
    np.random.normal(29, 7, n).round(), 18, 45
).astype(int)

gender = np.random.choice(
    ["Male", "Female"], n, p=[0.52, 0.48]
)

income = np.clip(
    np.random.lognormal(np.log(7_000_000), 0.65, n),
    2_000_000, 35_000_000
).round(-5).astype(int)

brands = np.random.choice(
    ["Apple", "Samsung", "Xiaomi/Redmi",
     "JBL", "Anker/Soundcore", "Other"],
    n,
    p=[0.12, 0.16, 0.24, 0.20, 0.14, 0.14]
)

awareness = np.random.binomial(1, 0.46, n)

def likert(x):
    return np.clip(np.rint(x), 1, 5).astype(int)

price = likert(
    3 + 0.18*(income > 10_000_000)
    - 0.12*(brands == "Xiaomi/Redmi")
    + np.random.normal(0, 0.9, n)
)

features = likert(
    3.25 + 0.28*awareness
    + np.random.normal(0, 0.85, n)
)

value = likert(
    2.9 + 0.22*awareness
    + 0.20*(price >= 4)
    + 0.10*(features >= 4)
    + np.random.normal(0, 0.9, n)
)

convenience = likert(
    3 + 0.15*awareness
    + np.random.normal(0, 0.95, n)
)

# ==========================================
# 2. GENERATE BRAND CONSIDERATION
# ==========================================

z = (
    -2 + 1.0*awareness
    + 0.18*(price-3)
    + 0.22*(features-3)
    + 0.32*(value-3)
    + 0.18*(convenience-3)
)

prob = 1 / (1 + np.exp(-z))

consideration = np.random.binomial(1, prob, n)

# ==========================================
# 3. GENERATE PURCHASE INTENTION
# ==========================================

latent = (
    -0.15 + 0.30*awareness
    + 0.18*(price-3)
    + 0.32*(features-3)
    + 0.48*(value-3)
    + 0.25*(convenience-3)
    + 0.65*consideration
    + np.random.normal(0, 1, n)
)

intention = np.digitize(
    latent, [-1.0, -0.25, 0.45, 1.2]
) + 1

# ==========================================
# 4. GENERATE MAIN PURCHASE BARRIER
# ==========================================

barriers = []

for a in awareness:
    if a == 0:
        barriers.append("Unaware of Sony TWS")
    else:
        barriers.append(np.random.choice(
            ["Price/value", "Feature mismatch",
             "Brand preference",
             "Availability/convenience",
             "No need/low urgency"],
            p=[0.32, 0.22, 0.20, 0.14, 0.12]
        ))

# ==========================================
# 5. CREATE DATAFRAME
# ==========================================

df = pd.DataFrame({
    "Respondent_ID": range(1, n+1),
    "Age": age,
    "Gender": gender,
    "Monthly_Income_IDR": income,
    "Current_Brand": brands,
    "Sony_Awareness": awareness,
    "Sony_Consideration": consideration,
    "Price_Perception": price,
    "Feature_Relevance": features,
    "Perceived_Value": value,
    "Purchase_Convenience": convenience,
    "Purchase_Intention": intention,
    "Purchase_Barrier": barriers
})

# Export dataset
df.to_csv("Sony_TWS_Indonesia_500.csv", index=False)

print("DATASET CREATED")
print(df.head(10))

# ==========================================
# 6. DESCRIPTIVE STATISTICS
# ==========================================

print("\nDESCRIPTIVE STATISTICS")

print(df.describe())

print("\nBRAND DISTRIBUTION")
print(df["Current_Brand"].value_counts(normalize=True)*100)

print("\nAWARENESS")
print(df["Sony_Awareness"].mean()*100)

print("\nCONSIDERATION")
print(df["Sony_Consideration"].mean()*100)

print("\nPURCHASE BARRIERS")
print(df["Purchase_Barrier"].value_counts(normalize=True)*100)

print("\nPURCHASE INTENTION")
print(df["Purchase_Intention"].value_counts(normalize=True)*100)

# ==========================================
# 7. MULTIPLE LINEAR REGRESSION
# ==========================================

X = df[[
    "Sony_Awareness",
    "Sony_Consideration",
    "Price_Perception",
    "Feature_Relevance",
    "Perceived_Value",
    "Purchase_Convenience"
]]

X = sm.add_constant(X)

Y = df["Purchase_Intention"]

model = sm.OLS(Y, X).fit()

print("\nREGRESSION RESULTS")
print(model.summary())

# ==========================================
# 8. AWARENESS VS PURCHASE INTENTION
# ==========================================

print("\nGROUP COMPARISON")

print(
    df.groupby("Sony_Awareness")
    .agg(
        Respondents=("Respondent_ID", "count"),
        Consideration=("Sony_Consideration", "mean"),
        Purchase_Intention=("Purchase_Intention", "mean")
    )
)

# ==========================================
# 9. HYPOTHESIS DECISION
# ==========================================

print("\nHYPOTHESIS TESTING")

for variable in X.columns[1:]:
    p = model.pvalues[variable]
    beta = model.params[variable]

    print(
        variable,
        "Beta:", round(beta, 3),
        "P-value:", round(p, 4),
        "Decision:",
        "Significant" if p < 0.05
        else "Not significant"
    )

print("\nCSV FILE EXPORTED SUCCESSFULLY")