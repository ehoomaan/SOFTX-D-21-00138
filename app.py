import io, re
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter

# --- Matplotlib: disable external LaTeX ---
mpl.rcParams["text.usetex"] = False
mpl.rcParams["mathtext.fontset"] = "dejavusans"
mpl.rcParams["font.family"] = "DejaVu Sans"

# --- pySigmaP imports ---
from pysigmap.data import Data
from pysigmap.casagrande import Casagrande
from pysigmap.pachecosilva import PachecoSilva
from pysigmap.boone import Boone
from pysigmap.bilog import Bilog
from pysigmap.energy import WangAndFrost, BeckerEtAl

st.set_page_config(page_title="pySigmaP GUI", layout="centered")
st.title("Preconsolidation Pressure (pySigmaP)")

st.write("""
Upload your oedometer CSV, pick a method, and compute sigma'p.
CSV must have **three columns** in this strict order:
1) stress (your units), 2) axial strain, 3) void ratio.
The first row should include the on-table void ratio.
""")

with st.expander("Try a sample dataset (authors' demo CSV)"):
    st.write("Use this URL if you want to test without your own file:")
    st.code("https://raw.githubusercontent.com/eamontoyaa/data4testing/main/pysigmap/testData.csv", language="text")

# ----------------- Units -----------------
UNITS = [
    "kPa", "MPa", "kN/m^2", "Pa",
    "psi", "psf", "ksf", "tsf",
    "bar", "kgf/cm^2"
]
TO_KPA = {
    "kPa": 1.0,
    "MPa": 1000.0,
    "kN/m^2": 1.0,          # 1 kN/m^2 = 1 kPa
    "Pa": 0.001,
    "bar": 100.0,
    "psi": 6.894757293168361,
    "psf": 0.04788025898033584,   # 1 psf = 47.88025898 Pa
    "ksf": 47.88025898033584,     # 1000 psf
    "tsf": 95.76051796067168,     # 2000 psf (short ton)
    "kgf/cm^2": 98.0665,
}

def to_kpa(x, unit):
    return float(x) * TO_KPA[unit]

def series_to_kpa(s, unit):
    return s.astype(float) * TO_KPA[unit]

def from_kpa(x_kpa, unit):
    return float(x_kpa) / TO_KPA[unit]

# --- Legend TeX sanitizer (handles titles like \textbf{...}) ---
def _strip_tex_macros(s: str) -> str:
    s = re.sub(r'\\textbf\{([^}]*)\}', r'\1', s)
    s = re.sub(r'\\mathbf\{([^}]*)\}', r'\1', s)
    s = s.replace(r'\,', ' ').replace(r'\;', ' ').replace(r'\:', ' ')
    return s

def fix_legend_tex(fig):
    for ax in fig.get_axes():
        leg = ax.get_legend()
        if not leg:
            continue
        title = leg.get_title()
        if title is not None:
            raw = title.get_text()
            clean = _strip_tex_macros(raw)
            if clean != raw:
                title.set_text(clean)
                title.set_fontweight('bold')
        for t in leg.get_texts():
            raw = t.get_text()
            clean = _strip_tex_macros(raw)
            if clean != raw:
                t.set_text(clean)
                t.set_fontweight('bold')

# --- Apply display units (tick labels + axis label + legend kPa->unit) ---
def apply_display_units(fig, unit):
    if unit == "kPa":
        return  # default already kPa
    factor = 1.0 / TO_KPA[unit]  # multiply kPa by factor to get target unit

    def fmt(val, pos=None):
        y = val * factor  # convert tick (in kPa) to target unit
        # compact formatting across log/linear
        if y == 0:
            return "0"
        # Use g-format; keep a couple of significant figs
        return f"{y:.6g}"

    for ax in fig.get_axes():
        # Ticks
        ax.xaxis.set_major_formatter(FuncFormatter(fmt))
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, p: ""))  # keep minors uncluttered

        # X label
        xlabel = ax.get_xlabel() or ""
        if "kPa" in xlabel:
            ax.set_xlabel(xlabel.replace("kPa", unit))
        elif xlabel.strip() != "":
            ax.set_xlabel(f"{xlabel} [{unit}]")
        else:
            ax.set_xlabel(f"Effective vertical stress, sigma'v [{unit}]")

        # Legend text: convert "... = <num> kPa" to the chosen unit
        leg = ax.get_legend()
        if leg:
            def repl(m):
                val = float(m.group(1))
                y = val * factor
                # choose a sensible format
                if y >= 100:
                    s = f"{y:.0f}"
                elif y >= 10:
                    s = f"{y:.1f}"
                else:
                    s = f"{y:.3g}"
                return f"{s} {unit}"

            for txt in leg.get_texts():
                s = txt.get_text()
                s2 = re.sub(r"([0-9]*\.?[0-9]+)\s*kPa", repl, s)
                s2 = s2.replace("[kPa]", f"[{unit}]")  # just in case
                if s2 != s:
                    txt.set_text(s2)

# ----------------- UI controls -----------------
uploaded = st.file_uploader("Upload CSV", type=["csv"])

# Unit selectors:
csv_unit = st.selectbox("Unit of stress in your CSV (first column)", UNITS, index=UNITS.index("kPa"))
sv_unit = st.selectbox("Unit for in-situ effective stress sigma'v0 input", UNITS, index=UNITS.index("kPa"))
display_unit = st.selectbox("Display results in", UNITS, index=UNITS.index("kPa"))

sigmaV_input = st.number_input(f"In-situ effective vertical stress sigma'v0 ({sv_unit})", min_value=0.0, value=75.0, step=1.0)

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

# ----------------- Main workflow -----------------
def run_analysis(df):
    # Convert stress column to kPa using chosen CSV unit
    dfk = df.copy()
    dfk.iloc[:, 0] = series_to_kpa(dfk.iloc[:, 0], csv_unit)

    # Convert in-situ input to kPa
    sigmaV_kpa = to_kpa(sigmaV_input, sv_unit)

    # Build data object (strain and void ratio columns untouched)
    data = Data(dfk.iloc[:, :3], sigmaV=sigmaV_kpa)

    # Show the raw curve first
    fig_curve = data.plot()
    fix_legend_tex(fig_curve)
    apply_display_units(fig_curve, display_unit)
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

    fix_legend_tex(fig)
    apply_display_units(fig, display_unit)

    st.subheader("Result")
    st.pyplot(fig)

    # Show numeric sigma'p if present
    sigma_p_val = None
    for name in ("sigmaP", "sigma_p", "sigmap", "SigmaP"):
        if hasattr(model, name):
            try:
                sigma_p_val = float(getattr(model, name))
                break
            except Exception:
                pass
    if sigma_p_val is not None:
        st.metric(f"Estimated sigma'p ({display_unit})", f"{from_kpa(sigma_p_val, display_unit):.3g}")

    # Download figure as PNG
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    st.download_button("Download figure (PNG)", data=buf.getvalue(), file_name="sigma_p.png", mime="image/png")

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        if df.shape[1] < 3:
            st.error("Need at least 3 columns: stress, axial strain, void ratio.")
        else:
            run_analysis(df)
    except Exception as e:
        st.exception(e)
else:
    st.info("Upload a CSV to begin. If using the demo file, set CSV unit to kPa.")
