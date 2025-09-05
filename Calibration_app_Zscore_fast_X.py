import streamlit as st
import pyodbc
import pandas as pd
import numpy as np
import os
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

# -------------------------------
# Database Config
# -------------------------------
DB_PASSWORD = "mpasecurity"

def make_conn_str(db_path):
    return (
        r"DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};"
        fr"DBQ={db_path};"
        fr"PWD={DB_PASSWORD};"
    )

# -------------------------------
# Cached DB Loaders
# -------------------------------
@st.cache_data(show_spinner=True)
def load_elements(conn_str):
    conn = pyodbc.connect(conn_str)
    df = pd.read_sql("SELECT * FROM setElementInfo", conn)
    conn.close()
    return df

@st.cache_data(show_spinner=True)
def load_concentrations(conn_str):
    conn = pyodbc.connect(conn_str)
    df = pd.read_sql("""
        SELECT M.SampleID, M.SampleName, M.SUS, M.MatID, D.ElementID, D.Conc
        FROM calSampleMaster M
        INNER JOIN calSampleDetails D ON M.SampleID = D.SampleID
        WHERE M.SUS <> 1
    """, conn)
    conn.close()
    return df

@st.cache_data(show_spinner=True)
def load_intensities(conn_str, analyte_id, base_id):
    conn = pyodbc.connect(conn_str)
    
    # Step 1: Get pixel IDs
    pixel_ids_df = pd.read_sql(
        f"SELECT ElementPixelID, ElementID FROM setElementPixelInfo WHERE ElementID IN ({analyte_id}, {base_id})",
        conn
    )
    pixel_ids = tuple(pixel_ids_df["ElementPixelID"].unique())
    pixel_ids_sql = str(pixel_ids) if len(pixel_ids) > 1 else f"({pixel_ids[0]})"
    
    # Step 2: Get intensity data
    query = f"""
        SELECT SampleID, ElementPixelID, Intensity, SparkNo
        FROM calData
        WHERE ElementPixelID IN {pixel_ids_sql}
    """
    df = pd.read_sql(query, conn)
    conn.close()
    
    # Add ElementID back
    df = df.merge(pixel_ids_df, on="ElementPixelID", how="left")
    return df

@st.cache_data(show_spinner=True)
def load_sd(conn_str):
    conn = pyodbc.connect(conn_str)
    df = pd.read_sql("SELECT * FROM libStdSample", conn)
    conn.close()
    return df

@st.cache_data(show_spinner=True)
def load_matrices(conn_str):
    conn = pyodbc.connect(conn_str)
    df = pd.read_sql("SELECT DISTINCT MatID FROM calSampleMaster", conn)
    conn.close()
    return df

# -------------------------------
# Fitting Function
# -------------------------------
def fit_model(analyte_pixel, base_pixel, spark, data, sd_df, conc_df, analyte_id, max_degree=2):
    results = []
    spark_data = data[data["SparkNo"] == spark].copy()

    if analyte_pixel not in spark_data or base_pixel not in spark_data:
        return results

    spark_data["IR"] = spark_data[analyte_pixel] / spark_data[base_pixel]
    spark_data = spark_data.replace([np.inf, -np.inf], np.nan).dropna(subset=["IR", "CR"])

    if spark_data.empty:
        return results

    X = spark_data["IR"].values
    y = spark_data["CR"].values

    # Merge SD safely
    merged = pd.merge(
        spark_data[["SampleID", "SampleName", "IR", "CR"]],
        sd_df[["SampleName", "SD"]],
        on="SampleName",
        how="left"
    )

    # Fallback for missing SD
    merged["SD"] = merged["SD"].replace(0, np.nan)
    if "Conc" not in merged:
        merged = merged.merge(
            conc_df[["SampleID", analyte_id]].rename(columns={analyte_id: "Conc"}),
            on="SampleID", how="left"
        )
    mask = merged["SD"].isna() | (merged["SD"] <= 0)
    merged.loc[mask, "SD"] = 0.0158 * (merged.loc[mask, "Conc"] ** 0.65)
    merged["SD"] = merged["SD"].replace(0, 1e-6)

    sd_vals = merged["SD"].values

    for deg in range(1, max_degree + 1):
        try:
            coeffs = np.polyfit(X, y, deg)
            y_pred = np.polyval(coeffs, X)

            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            residuals = y - y_pred
            z_scores = residuals / sd_vals
            mean_z = np.mean(np.abs(z_scores))

            results.append({
                "AnalytePixel": analyte_pixel,
                "BasePixel": base_pixel,
                "SparkNo": spark,
                "Degree": deg,
                "R2": r2,
                "MeanZ": mean_z
            })
        except Exception:
            continue

    return results

# -------------------------------
# Main App
# -------------------------------
st.set_page_config(page_title="⚡ Fast Calibration Fit Finder", layout="wide")
st.title("⚡ Calibration Best Fit Finder (with Matrix & Safe Z-Score)")

# DB path input
db_path = st.text_input(
    "Enter full path to Access Database (.mdb/.accdb):",
    value=r"C:\Users\mpa1326\Calibration\Data\MPAAnalystDB.mdb"
)

if db_path and os.path.exists(db_path):
    conn_str = make_conn_str(db_path)

    elem_df = load_elements(conn_str)
    conc_df = load_concentrations(conn_str)
    sd_df = load_sd(conn_str)
    mat_df = load_matrices(conn_str)

    st.success("✅ Database connected successfully!")

    # --- UI selections ---
    matrix_id = st.selectbox("Select Matrix", mat_df["MatID"].unique())
    base_symbol = st.selectbox("Select Base Element", elem_df["EleSymbol"].unique())
    analyte_symbol = st.selectbox("Select Analyte Element", elem_df["EleSymbol"].unique())
    max_degree = st.slider("Max Polynomial Degree", 1, 3, 2)

    if st.button("Run Analysis"):
        # Map symbols to IDs
        base_id = elem_df.loc[elem_df["EleSymbol"] == base_symbol, "ElementID"].iloc[0]
        analyte_id = elem_df.loc[elem_df["EleSymbol"] == analyte_symbol, "ElementID"].iloc[0]

        # Filter by matrix
        conc_df = conc_df[conc_df["MatID"] == matrix_id]

        st.info("📥 Loading intensity data...")
        intens_df = load_intensities(conn_str, analyte_id, base_id)

        # Pivot intensities
        intens_pivot = intens_df.pivot_table(
            index=["SampleID", "SparkNo"],
            columns="ElementPixelID",
            values="Intensity",
            aggfunc="mean"
        )

        # Pivot concentrations
        conc_pivot = conc_df.pivot_table(
            index=["SampleID", "SampleName"],
            columns="ElementID",
            values="Conc",
            aggfunc="mean"
        )

        if analyte_id in conc_pivot and base_id in conc_pivot:
            conc_pivot["CR"] = conc_pivot[analyte_id] / conc_pivot[base_id]
            conc_pivot = conc_pivot.reset_index()

            merged = pd.merge(intens_pivot.reset_index(), conc_pivot, on="SampleID", how="inner")
        else:
            merged = pd.DataFrame()

        analyte_pixels = intens_df.loc[intens_df["ElementID"] == analyte_id, "ElementPixelID"].unique()
        base_pixels = intens_df.loc[intens_df["ElementID"] == base_id, "ElementPixelID"].unique()
        sparks = intens_df["SparkNo"].unique()

        st.write(f"🔎 Testing {len(analyte_pixels)} analyte pixels × {len(base_pixels)} base pixels × {len(sparks)} sparks")

        # Parallel fitting
        results = Parallel(n_jobs=-1, verbose=0)(
            delayed(fit_model)(ap, bp, sp, merged, sd_df, conc_pivot, analyte_id, max_degree)
            for ap in analyte_pixels
            for bp in base_pixels
            for sp in sparks
        )

        results_flat = [item for sublist in results for item in sublist]
        results_df = pd.DataFrame(results_flat)

        if results_df.empty:
            st.warning("⚠️ No valid fits found.")
        else:
            sort_by = st.radio("Sort results by:", ["R2", "MeanZ"])
            ascending = (sort_by == "MeanZ")
            results_df = results_df.sort_values(by=sort_by, ascending=ascending)

            st.dataframe(results_df)

            # Plot best fit
            best = results_df.iloc[0]
            st.subheader("📈 Best Fit Curve")
            st.write(best)

            spark_data = merged[merged["SparkNo"] == best["SparkNo"]]
            spark_data["IR"] = spark_data[best["AnalytePixel"]] / spark_data[best["BasePixel"]]

            coeffs = np.polyfit(spark_data["IR"].values, spark_data["CR"].values, int(best["Degree"]))
            x_fit = np.linspace(spark_data["IR"].min(), spark_data["IR"].max(), 100)
            y_fit = np.polyval(coeffs, x_fit)

            fig, ax = plt.subplots()
            ax.scatter(spark_data["IR"], spark_data["CR"], label="Data", color="blue")
            ax.plot(x_fit, y_fit, label=f"Poly Deg {best['Degree']}", color="red")
            ax.set_xlabel("IR (Intensity Ratio)")
            ax.set_ylabel("CR (Conc Ratio)")
            ax.legend()
            st.pyplot(fig)

else:
    st.info("📂 Please enter a valid database path.")
