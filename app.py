import io
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl
# --- put this near the top of app.py ---
import re
import matplotlib as mpl

def _strip_tex_macros(s: str) -> str:
    # Remove \textbf{...} and \mathbf{...} while keeping the content
    s = re.sub(r'\\textbf\{([^}]*)\}', r'\1', s)
    s = re.sub(r'\\mathbf\{([^}]*)\}', r'\1', s)
    # Optional: clean a few harmless TeX tokens if they ever appear
    s = s.replace(r'\,', ' ').replace(r'\;', ' ').replace(r'\:', ' ')
    return s

def fix_legend_tex(fig):
    # Go through every axes and sanitize legend text/title
    for ax in fig.get_axes():
        leg = ax.get_legend()
        if not leg:
            continue
        # Title
        title = leg.get_title()
        if title is not None:
            raw = title.get_text()
            clean = _strip_tex_macros(raw)
            if clean != raw:
                title.set_text(clean)
                title.set_fontweight('bold')
        # Entries
        for t in leg.get_texts():
            raw = t.get_text()
            clean = _strip_tex_macros(raw)
            if clean != raw:
                t.set_text(clean)
                t.set_fontweight('bold')

# IMPORTANT: Disable LaTeX so it works on Streamlit Cloud (no TeX installed there)
mpl.rcParams["text.usetex"] = False
mpl.rcParams["mathtext.fontset"] = "dejavusans"
mpl.rcParams["font.family"] = "DejaVu Sans"

from pysigmap.data import Data
from pysigmap.casagrande import Casagrande
from pysigmap.pachecosilva import PachecoSilva
from pysigmap.boone import Boone
from pysigmap.bilog import Bilog
from pysigmap.energy import WangAndFrost, BeckerEtAl

st.set_page_config(page_title="pySigmaP GUI", layout="centered")

st.title("Preconsolidation Pressure (pySigmaP)")
st.write("Upload your oedometer CSV, pick a method, and compute σ'ₚ. "
         "CSV must have **three columns** in this strict order: "
         "effective vertical stress σ'v (kPa), axial strain, void ratio. "
         "The first row should include the on-table void ratio.")

with st.expander("Try a sample dataset (authors' demo CSV)"):
    st.write("Use this URL in case you want to test without your own file:")
    st.code("https://raw.githubusercontent.com/eamontoyaa/data4testing/main/pysigmap/testData.csv", language="text")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
sigmaV = st.number_input("In-situ effective vertical stress σ'v0 (kPa)", min_value=0.0, value=75.0, step=1.0)

method_name = st.selectbox(
    "Method",
    [
        "Casagrande",
        "Pacheco Silva",
        "Boone",
        "Butterfield (bilog)",
        "Oikawa (bilog)",
        "Onitsuka et al. (bilog)",
        "Wang & Frost (energy)",
        "Becker et al. (energy)",
    ],
)

def run_analysis(df):
    data = Data(df.iloc[:, :3], sigmaV=sigmaV)

    # Show the raw curve first
    fig_curve = data.plot()
    st.pyplot(fig_curve)

    # Choose and run the method
    if method_name == "Casagrande":
        model = Casagrande(data)
        fig = model.getSigmaP(loglog=True)
    elif method_name == "Pacheco Silva":
        model = PachecoSilva(data)
        fig = model.getSigmaP()
    elif method_name == "Boone":
        model = Boone(data)
        fig = model.getSigmaP()
    elif method_name.startswith("Butterfield"):
        model = Bilog(data)
        fig = model.getSigmaP(opt=1)  # Butterfield
    elif method_name.startswith("Oikawa"):
        model = Bilog(data)
        fig = model.getSigmaP(opt=2)  # Oikawa
    elif method_name.startswith("Onitsuka"):
        model = Bilog(data)
        fig = model.getSigmaP(opt=3)  # Onitsuka et al.
    elif method_name.startswith("Wang"):
        model = WangAndFrost(data)
        fig = model.getSigmaP()
    else:
        model = BeckerEtAl(data)
        fig = model.getSigmaP()

    st.subheader("Result")
    st.pyplot(fig)

    # Download figure as PNG
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    st.download_button("Download figure (PNG)", data=buf.getvalue(), file_name="sigma_p.png", mime="image/png")

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        if df.shape[1] < 3:
            st.error("Need at least 3 columns: σ'v, axial strain, void ratio.")
        else:
            run_analysis(df)
    except Exception as e:
        st.exception(e)
else:
    st.info("Upload a CSV to begin, or copy the demo URL above into your browser and download the file.")
