import streamlit as st
import pyodbc
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import os

DB_PASSWORD = "mpasecurity"

def make_conn_str(db_path):
    return (
        r"DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};"
        fr"DBQ={db_path};"
        fr"PWD={DB_PASSWORD};"
    )

# --- Core Function ---
def get_best_fit(conn_str, base_symbol, analyte_symbol, poly_degrees=[1, 2]):
    conn = pyodbc.connect(conn_str)

    # Element IDs
    elem_df = pd.read_sql("SELECT * FROM setElementInfo", conn)
    base_id = elem_df.loc[elem_df["EleSymbol"] == base_symbol, "ElementID"].iloc[0]
    analyte_id = elem_df.loc[elem_df["EleSymbol"] == analyte_symbol, "ElementID"].iloc[0]

    # Concentration info
    conc_q = f"""
    SELECT M.SampleID, M.SampleName, M.SUS, D.ElementID, D.Conc
    FROM calSampleMaster AS M
    INNER JOIN calSampleDetails AS D ON M.SampleID = D.SampleID
    WHERE M.SUS <> 1 AND D.ElementID IN ({base_id}, {analyte_id})
    """
    conc_df = pd.read_sql(conc_q, conn)
    conc_pivot = conc_df.pivot_table(index=["SampleID","SampleName"],
                                     columns="ElementID", values="Conc").reset_index()
    conc_pivot["CR"] = conc_pivot[analyte_id] / conc_pivot[base_id]

    # SD info from libStdSample (for analyte only)
    sd_q = f"""
    SELECT SampleName, ElementID, Conc, SD
    FROM libStdSample
    WHERE ElementID = {analyte_id}
    """
    sd_df = pd.read_sql(sd_q, conn)

    # Replace NaN or 0 with formula
    sd_df["SD"] = sd_df.apply(
        lambda row: 0.0158 * (row["Conc"] ** 0.65)
        if pd.isna(row["SD"]) or row["SD"] == 0
        else row["SD"],
        axis=1
    )


    # Pixel info
    pixel_df = pd.read_sql("SELECT * FROM setElementPixelInfo", conn)
    analyte_pixels = pixel_df.loc[pixel_df["ElementID"] == analyte_id, "ElementPixelID"].tolist()
    base_pixels = pixel_df.loc[pixel_df["ElementID"] == base_id, "ElementPixelID"].tolist()

    # Intensities
    intens_q = f"""
    SELECT SampleID, ElementPixelID, SparkNo, Intensity
    FROM calData
    WHERE ElementPixelID IN ({",".join(map(str, analyte_pixels + base_pixels))})
    """
    intens_df = pd.read_sql(intens_q, conn)
    conn.close()

    intens_pivot = intens_df.pivot_table(index=["SampleID","SparkNo"],
                                         columns="ElementPixelID", values="Intensity").reset_index()

    data = pd.merge(intens_pivot, conc_pivot[["SampleID","CR"]], on="SampleID")

    results = []
    for ap in analyte_pixels:
        for bp in base_pixels:
            if ap in data.columns and bp in data.columns:
                for spark in sorted(data["SparkNo"].unique()):
                    spark_data = data[data["SparkNo"] == spark].copy()
                    spark_data["IR"] = spark_data[ap] / spark_data[bp]

                    # Drop NaN/Inf
                    spark_data = spark_data.replace([np.inf, -np.inf], np.nan).dropna(subset=["CR","IR"])
                    if spark_data.empty:
                        continue

                    X = spark_data[["IR"]].values
                    y = spark_data["CR"].values

                    for deg in poly_degrees:
                        try:
                            model = make_pipeline(PolynomialFeatures(degree=deg), LinearRegression())
                            model.fit(X, y)
                            r2 = model.score(X, y)

                            # Predict and compute Z-score
                            y_pred = model.predict(X)
                            residuals = y - y_pred

                            # Attach SampleName to spark_data
                            spark_data_named = pd.merge(
                                spark_data,
                                conc_pivot[["SampleID", "SampleName"]],
                                on="SampleID",
                                how="left"
                            )

                            # Merge with libStdSample on SampleName
                            merged = pd.merge(
                                spark_data_named,
                                sd_df,
                                on="SampleName",
                                how="left"
                            )

                            if merged["SD"].isna().all():
                                mean_z = np.nan
                            else:
                                # Avoid division by zero: replace 0 or NaN with max SD or 1
                                sd_vals = merged["SD"].replace(0, np.nan)
                                sd_vals = sd_vals.fillna(sd_vals.max()).fillna(1).values
                                z_scores = residuals / sd_vals
                                mean_z = np.mean(np.abs(z_scores))

                            results.append({
                                "AnalytePixel": ap,
                                "BasePixel": bp,
                                "SparkNo": spark,
                                "Degree": deg,
                                "R2": r2,
                                "ZScore": mean_z,
                                "X": X.flatten(),
                                "y": y,
                                "model": model
                            })
                        except Exception:
                            continue

    results_df = pd.DataFrame(results)
    if results_df.empty:
        return pd.DataFrame(), None

    results_df = results_df.sort_values("R2", ascending=False)
    best = results_df.iloc[0]
    return results_df, best


# --- Streamlit UI ---
st.set_page_config(page_title="Calibration Best Fit Finder", layout="wide")
st.title("🔬 Calibration Best Fit Finder")

# DB Path input
db_path = st.text_input(
    "Enter full path to Access Database (.mdb / .accdb)",
    value=r"C:\Users\mpa1326\Calibration\Data\MPAAnalystDB.mdb"
)

if db_path:
    if not os.path.exists(db_path):
        st.error("❌ Database file not found. Please check the path.")
    else:
        conn_str = make_conn_str(db_path)
        try:
            conn = pyodbc.connect(conn_str)
            elem_df = pd.read_sql("SELECT EleSymbol FROM setElementInfo", conn)
            conn.close()

            st.success("✅ Database connected successfully!")

            col1, col2 = st.columns(2)
            with col1:
                base_symbol = st.selectbox("Select Base Element", elem_df["EleSymbol"].unique())
            with col2:
                analyte_symbol = st.selectbox("Select Analyte Element", elem_df["EleSymbol"].unique())

            if st.button("Run Analysis"):
                start_time = time.time()
                with st.spinner("Processing..."):
                    results_df, best = get_best_fit(conn_str, base_symbol, analyte_symbol)
                elapsed = time.time() - start_time

                if results_df.empty:
                    st.error("⚠️ No valid results found. Check data availability.")
                else:
                    st.success(f"✅ Analysis completed in {elapsed:.2f} seconds")

                    sort_by = st.radio("Sort results by:", ["R2", "ZScore"])
                    if sort_by == "R2":
                        results_df = results_df.sort_values("R2", ascending=False)
                    else:
                        results_df = results_df.sort_values("ZScore", ascending=True)

                    st.subheader("📊 Top Results")
                    st.dataframe(results_df[["AnalytePixel","BasePixel","SparkNo","Degree","R2","ZScore"]].head(20))

                    best = results_df.iloc[0]
                    st.subheader("🏆 Best Combination")
                    st.write(best)

                    # Plot best fit
                    X = best["X"]
                    y = best["y"]
                    model = best["model"]
                    x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
                    y_line = model.predict(x_line)

                    fig, ax = plt.subplots(figsize=(7,5))
                    ax.scatter(X, y, color="blue", label="Data Points")
                    ax.plot(x_line, y_line, color="red", linewidth=2,
                            label=f"Best Fit (deg={best['Degree']}, R²={best['R2']:.3f}, Z={best['ZScore']:.3f})")
                    ax.set_xlabel("Intensity Ratio (IR)")
                    ax.set_ylabel("Concentration Ratio (CR)")
                    ax.set_title(f"Best Fit for {analyte_symbol}/{base_symbol}, Spark {best['SparkNo']}")
                    ax.legend()
                    ax.grid(True, linestyle="--", alpha=0.6)
                    st.pyplot(fig)

        except Exception as e:
            st.error(f"⚠️ Error connecting to database: {e}")
else:
    st.info("📂 Please enter path to your Access DB.")
