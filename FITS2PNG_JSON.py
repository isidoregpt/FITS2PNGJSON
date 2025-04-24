import os
import json
import numpy as np
import shutil
from astropy.io import fits
from astropy.time import Time
from astropy.visualization import astropy_mpl_style, ZScaleInterval
import matplotlib.pyplot as plt
import streamlit as st

# ——— Utility functions (ported from your script) ———

def header_to_dict(header):
    d = {}
    for card in header.cards:
        key = card.keyword
        if not key or key in ("COMMENT", "HISTORY"):
            continue
        val = card.value
        if isinstance(val, np.generic):
            val = val.item()
        d[key] = val
    return d

def extract_comments(header):
    comments = {}
    for card in header.cards:
        if card.keyword and card.comment:
            comments[card.keyword] = card.comment
    return comments

def extract_metadata(header, data):
    meta = {}
    meta["header"] = header_to_dict(header)
    comm = extract_comments(header)
    if comm:
        meta["header_comments"] = comm

    # observation time
    for k in ("DATE-OBS", "DATE_OBS", "DATE", "OBSDATE"):
        if k in header:
            try:
                meta["observation_time"] = Time(header[k]).iso
                break
            except Exception:
                pass

    # solar-disk params
    sun_keys = ("FNDLMBXC", "FNDLMBYC", "FNDLMBMI", "FNDLMBMA")
    if all(k in header for k in sun_keys):
        try:
            cx = float(header["FNDLMBXC"])
            cy = float(header["FNDLMBYC"])
            minor = float(header["FNDLMBMI"])
            major = float(header["FNDLMBMA"])
            meta["sun_params"] = {
                "cx": cx,
                "cy": cy,
                "radius": (minor + major) / 4.0
            }
        except Exception:
            pass
    if "sun_params" not in meta:
        h, w = data.shape
        meta["sun_params"] = {"cx": w/2.0, "cy": h/2.0, "radius": min(w, h)*0.45}

    meta["data_shape"] = list(data.shape)
    meta["dtype"] = str(data.dtype)
    meta["data_stats"] = {
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "std": float(np.std(data)),
    }
    return meta

def render_png(data, out_path, dpi=300, figsize=(8, 8)):
    plt.style.use(astropy_mpl_style)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    vmin, vmax = ZScaleInterval().get_limits(data)
    plt.imshow(data, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    plt.axis("off")
    plt.tight_layout(pad=0)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

# ——— Streamlit UI ———

st.set_page_config(
    page_title="FITS → PNG/JSON Converter",
    layout="wide",
)

st.title("🔭 FITS → PNG & JSON Converter")
st.markdown("""
This tool transforms astronomical FITS files into:
1. **PNG** images (z-scaled, publication-quality).  
2. **JSON** metadata (headers, comments, timestamps, sun parameters, basic stats).

**Steps to use:**
- **Upload** one or more `.fit` or `.fits` files below.  
- **(Optional)** Change the output directory in the sidebar.  
- Click **Convert** and watch the progress bar.  
- Download your results as a ZIP or inspect the individual files in the output folder.
""")

# Sidebar settings
st.sidebar.header("⚙️ Settings")
default_out = os.path.abspath("Output")
output_dir = st.sidebar.text_input("Output directory", value=default_out)
if not os.path.exists(output_dir):
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        st.sidebar.error(f"Could not create folder: {e}")

# File uploader
uploaded = st.file_uploader(
    "Upload FITS files",
    type=["fit", "fits"],
    accept_multiple_files=True
)

# Convert button
if uploaded:
    if st.button("▶️ Convert"):
        total = len(uploaded)
        progress = st.progress(0)
        status = st.empty()

        for i, file in enumerate(uploaded, start=1):
            fname = file.name
            status.text(f"Processing **{fname}** ({i}/{total})…")
            try:
                hdul = fits.open(file)
                data = np.nan_to_num(hdul[0].data)
                header = hdul[0].header
                hdul.close()

                # metadata + rendering
                meta = extract_metadata(header, data)
                name, _ = os.path.splitext(fname)

                png_path = os.path.join(output_dir, f"{name}.png")
                render_png(data, png_path)

                json_path = os.path.join(output_dir, f"{name}.json")
                with open(json_path, "w") as jf:
                    json.dump(meta, jf, indent=2)

                status.success(f"✔ {fname} → `{name}.png` + `{name}.json`")
            except Exception as e:
                status.error(f"✖ Error on {fname}: {e}")

            progress.progress(i / total)

        # ZIP & download
        zip_base = os.path.join(output_dir, "converted")
        shutil.make_archive(base_name=zip_base, format="zip", root_dir=output_dir)
        zip_path = zip_base + ".zip"
        with open(zip_path, "rb") as f:
            st.download_button(
                label="📦 Download all results as ZIP",
                data=f,
                file_name="fits_converted_results.zip",
                mime="application/zip"
            )

        st.balloons()
        st.success(f"Conversion complete! Files written to:\n`{output_dir}`")
else:
    st.info("Upload FITS files to get started.")

