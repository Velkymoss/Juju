import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadstat
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import gaussian_kde, pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


df, meta = pyreadstat.read_sav("data.sav")

# krank binär
df["krank_binary"] = df["Krank_JaNein"].replace({2: 1, 3: 1})

# Immunsuppression
df["Immunsuppression_binary"] = df[
    "Query_Immunolg_M_W_E.Immunsuppression0nein1ja"
].replace({np.nan: 0})

# S08
IQR = df["S08"].quantile(0.75) - df["S08"].quantile(0.25)
Q1 = df["S08"].quantile(0.25)
Q3 = df["S08"].quantile(0.75)
outliers = df[(df["S08"] < Q1 - 1.5 * IQR) | (df["S08"] > Q3 + 1.5 * IQR)]
outlier_mask = (df["S08"] < Q1 - 1.5 * IQR) | (df["S08"] > Q3 + 1.5 * IQR)
df = df[~outlier_mask]

# year
date_column = "dat_labor"

if date_column in df.columns:
    # Ensure the column is datetime
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")

    # Remove any NaT (Not a Time) values for analysis
    date_data = df[date_column].dropna()

    # Create 2 plots side by side
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 1. Distribution by month
    date_data.dt.to_period("M").value_counts().sort_index().plot(kind="bar", ax=axes[0])
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("Number of Records")
    axes[0].set_title("Data Distribution by Month")
    axes[0].tick_params(axis="x", rotation=45)

    # 2. Distribution by year
    date_data.dt.year.value_counts().sort_index().plot(kind="bar", ax=axes[1])
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Number of Records")
    axes[1].set_title("Data Distribution by Year")
    axes[1].tick_params(axis="x", rotation=0)

    plt.tight_layout()
    plt.savefig("plots/year.png")
    plt.show()

else:
    print(f"Column '{date_column}' not found. Please check the column name.")

df["year"] = df["dat_labor"].dt.year  # .astype('Int64')
df = df.dropna(subset=["year"])
df["year"] = df["year"].astype(int)
for y in [2021, 2022, 2023]:
    df[str(y)] = df["year"] == y

# age
df = df[df["AGE"] >= 60]

# barthelindex
df = df[df["BARTHELINDEX"] <= 100]

# Anzahl Imp Baseline
df["anzahl_Imp_baseline_binary"] = df["anzahl_Imp_baseline"].replace(
    {1: 0, 2: 1, 3: 1, 4: 1, 5: 1}
)
print(df["anzahl_Imp_baseline_binary"].value_counts())

# Anzahl Inf Baseline
# NC0422_pn, wenn > 0.422, dann inf baseline positiv, 25 leute ~
df["anzahl_Inf_baseline"] = df["anzahl_Inf_baseline"].replace({np.nan: 0})
df["anzahl_Inf_baseline"] = df["anzahl_Inf_baseline"].replace({2: 1, 3: 1, 4: 1})
print(df["anzahl_Inf_baseline"].value_counts())
df.loc[df["NC0422"] > 0.422, "anzahl_Inf_baseline"] = 1
print(df["anzahl_Inf_baseline"].value_counts())

# grundimmunisiert
df["grundimmunisiert"] = (
    (df["anzahl_Inf_baseline"] == 1) & (df["anzahl_Imp_baseline"] >= 2)
).astype(int)


########################################### Correlation matrix############################
correlation_columns = [
    "S08",
    "SEX",
    "AGE",
    "BMI",
    "MOCA",
    "BARTHELINDEX",
    "anzahl_Imp_baseline",
    "anzahl_Inf_baseline",
    "Nursing_Home",
    "Immunsuppression_binary",
    "krank_binary",
    "grundimmunisiert",
    "anzahl_Imp_baseline_binary",
]

# Define which variables are categorical/ordinal (use Spearman)
categorical_vars = [
    "anzahl_Imp_baseline",
    "anzahl_Inf_baseline",
    "SEX",
    "Nursing_Home",
    "Immunsuppression_binary",
    "krank_binary",
    "grundimmunisiert",
    "anzahl_Imp_baseline_binary",
]

# Get existing columns (include all types, not just numeric)
existing_columns = [col for col in correlation_columns if col in df.columns]
corr_data = df[existing_columns].copy()

# Convert categorical to numeric if needed
for col in corr_data.columns:
    if corr_data[col].dtype == "object":
        corr_data[col] = pd.to_numeric(corr_data[col], errors="coerce")

if len(corr_data.columns) > 1:
    # Calculate correlation matrix and p-values
    n_vars = len(corr_data.columns)
    corr_matrix = np.zeros((n_vars, n_vars))
    p_matrix = np.zeros((n_vars, n_vars))
    method_matrix = np.full((n_vars, n_vars), "", dtype=object)

    for i in range(n_vars):
        for j in range(n_vars):
            col1 = corr_data.columns[i]
            col2 = corr_data.columns[j]

            if i == j:
                corr_matrix[i, j] = 1.0
                p_matrix[i, j] = 0.0
                method_matrix[i, j] = "diag"
            else:
                # Remove NaN values for each pair
                x = corr_data.iloc[:, i].dropna()
                y = corr_data.iloc[:, j].dropna()

                # Find common indices (both variables have values)
                common_idx = x.index.intersection(y.index)
                if len(common_idx) > 2:  # Need at least 3 observations
                    # Choose method based on variable types
                    if col1 in categorical_vars or col2 in categorical_vars:
                        # Use Spearman for categorical/ordinal variables
                        corr, p_val = spearmanr(x[common_idx], y[common_idx])
                        method_matrix[i, j] = "S"
                    else:
                        # Use Pearson for continuous variables
                        corr, p_val = pearsonr(x[common_idx], y[common_idx])
                        method_matrix[i, j] = "P"

                    corr_matrix[i, j] = corr
                    p_matrix[i, j] = p_val
                else:
                    corr_matrix[i, j] = np.nan
                    p_matrix[i, j] = np.nan
                    method_matrix[i, j] = "NA"

    # Convert to DataFrames
    corr_df = pd.DataFrame(
        corr_matrix, index=corr_data.columns, columns=corr_data.columns
    )
    p_df = pd.DataFrame(p_matrix, index=corr_data.columns, columns=corr_data.columns)
    method_df = pd.DataFrame(
        method_matrix, index=corr_data.columns, columns=corr_data.columns
    )

    # Create significance mask (p < 0.05)
    sig_mask = p_df < 0.05

    # Plot correlation matrix with significance asterisks and method indicators
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_df, dtype=bool))

    # Create annotations with significance markers only
    annot_matrix = corr_df.round(3).astype(str)
    for i in range(len(corr_df.columns)):
        for j in range(len(corr_df.columns)):
            if not mask[i, j]:  # Only for lower triangle
                p_val = p_df.iloc[i, j]

                # Add significance markers only
                if p_val < 0.001:
                    annot_matrix.iloc[i, j] += "***"
                elif p_val < 0.01:
                    annot_matrix.iloc[i, j] += "**"
                elif p_val < 0.05:
                    annot_matrix.iloc[i, j] += "*"

    sns.heatmap(
        corr_df,
        annot=annot_matrix,
        fmt="",
        cmap="RdBu_r",
        center=0,
        square=True,
        mask=mask,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 12},
    )

    plt.title(
        "Mixed Correlation Matrix\n* p<0.05, ** p<0.01, *** p<0.001",
        fontsize=14,
        pad=20,
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("plots/correlation_matrix.png")
    plt.show()

    for i, col1 in enumerate(corr_df.columns):
        for j, col2 in enumerate(corr_df.columns):
            if i > j:  # Only lower triangle
                corr_val = corr_df.iloc[i, j]
                p_val = p_df.iloc[i, j]
                method = method_df.iloc[i, j]

                sig_level = ""
                if p_val < 0.001:
                    sig_level = " (***)"
                elif p_val < 0.01:
                    sig_level = " (**)"
                elif p_val < 0.05:
                    sig_level = " (*)"

                method_name = (
                    "Pearson"
                    if method == "P"
                    else "Spearman"
                    if method == "S"
                    else "N/A"
                )
else:
    print("Not enough numeric columns for correlation analysis")

###################################################################################################

################################### Regression ###################################################
dependent_var = "S08"
independent_vars = [
    "SEX",
    "AGE",
    "BMI",
    "MOCA",
    "BARTHELINDEX",
    #"anzahl_Imp_baseline",
    "anzahl_Inf_baseline",
    "Nursing_Home",
    "Immunsuppression_binary",
    "krank_binary",
    "2022",
    "2023",
    "grundimmunisiert",
    "anzahl_Imp_baseline_binary",
]

# Filter data to include only complete cases
regression_data = df[[dependent_var] + independent_vars].dropna()

print(f"Sample size for regression: {len(regression_data)}")
print(f"Variables included: {independent_vars}")

if (
    len(regression_data) > len(independent_vars) + 1
):  # Need more observations than variables
    y = regression_data[dependent_var]

    # Make sure predictors are numeric
    X = regression_data[independent_vars].apply(pd.to_numeric, errors="coerce")

    # Add constant term (intercept)
    X_with_const = sm.add_constant(X)
    X_with_const = X_with_const.astype(float)

    # ✅ Compute VIF
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    vif_data = pd.DataFrame()
    vif_data["Variable"] = X_with_const.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X_with_const.values, i)
        for i in range(X_with_const.shape[1])
    ]

    print("\n" + "=" * 60)
    print("VARIANCE INFLATION FACTORS (VIF)")
    print("=" * 60)
    print(vif_data)

    # Fit the regression model
    model = sm.OLS(y, X_with_const).fit()

    # Print comprehensive results
    # print("\n" + "=" * 80)
    # print("LINEAR REGRESSION RESULTS")
    # print("=" * 80)
    # print(model.summary())

    # Create detailed results table
    results_df = pd.DataFrame(
        {
            "Variable": ["Intercept"] + independent_vars,
            "Coefficient": model.params.values,
            "Std_Error": model.bse.values,
            "t_value": model.tvalues.values,
            "p_value": model.pvalues.values,
            "CI_Lower": model.conf_int()[0].values,
            "CI_Upper": model.conf_int()[1].values,
        }
    )

    # Calculate standardized coefficients (Beta coefficients)
    # Standardize X variables (not the constant)
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)
    y_standardized = (y - y.mean()) / y.std()

    # Fit model with standardized variables
    X_std_with_const = sm.add_constant(X_standardized)
    model_std = sm.OLS(y_standardized, X_std_with_const).fit()

    # Add standardized coefficients (skip intercept for standardized)
    results_df["Beta_Standardized"] = [np.nan] + list(model_std.params.values[1:])

    # Add significance indicators
    def significance_stars(p_val):
        if p_val < 0.001:
            return "***"
        elif p_val < 0.01:
            return "**"
        elif p_val < 0.05:
            return "*"
        elif p_val < 0.1:
            return "."
        else:
            return ""

    results_df["Significance"] = results_df["p_value"].apply(significance_stars)

    # Display detailed results table
    print("\n" + "=" * 120)
    print("DETAILED REGRESSION COEFFICIENTS TABLE")
    print("=" * 120)

    # Format the table nicely
    pd.set_option("display.float_format", "{:.4f}".format)
    print(results_df.round(4))

    print("\nSignificance codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")

    # Model diagnostics
    print("\n" + "=" * 60)
    print("MODEL DIAGNOSTICS")
    print("=" * 60)
    print(f"R-squared: {model.rsquared:.4f}")
    print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
    print(f"F-statistic: {model.fvalue:.4f}")
    print(f"F-statistic p-value: {model.f_pvalue:.4e}")
    print(f"AIC: {model.aic:.2f}")
    print(f"BIC: {model.bic:.2f}")
    print(f"Log-Likelihood: {model.llf:.2f}")
    print(f"Durbin-Watson: {sm.stats.durbin_watson(model.resid):.4f}")

    residuals = model.resid
    fitted_values = model.fittedvalues

    # Create residual plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Q-Q plot
    sm.qqplot(residuals, line="s", ax=axes[0])
    axes[0].set_title("Q-Q Plot of Residuals")

    # Histogram of residuals with KDE
    axes[1].hist(
        residuals,
        bins=20,
        alpha=0.7,
        edgecolor="black",
        density=True,
        label="Histogram",
    )

    # Add KDE

    kde = gaussian_kde(residuals)
    x_range = np.linspace(residuals.min(), residuals.max(), 100)
    axes[1].plot(x_range, kde(x_range), "red", linewidth=2, label="KDE")

    axes[1].set_xlabel("Residuals")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Distribution of Residuals")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("plots/residuals.png")
    plt.show()

else:
    print("❌ Not enough data points for regression analysis")
    print(
        f"Need at least {len(independent_vars) + 2} observations, have {len(regression_data)}"
    )
