import os
import json
import numpy as np
import shutil
import time
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
            except Exception as e:
                st.warning(f"Could not parse time from {k}: {e}")
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
        except Exception as e:
            st.warning(f"Could not extract sun params: {e}")
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
    try:
        plt.style.use(astropy_mpl_style)
        fig = plt.figure(figsize=figsize, dpi=dpi)
        vmin, vmax = ZScaleInterval().get_limits(data)
        plt.imshow(data, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        plt.axis("off")
        plt.tight_layout(pad=0)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        return True
    except Exception as e:
        st.error(f"Error rendering PNG: {e}")
        return False

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

# Debug info
st.sidebar.subheader("Debug Info")
st.sidebar.info(f"Current working directory: {os.getcwd()}")
st.sidebar.info(f"Output directory: {output_dir}")
st.sidebar.info(f"Output directory exists: {os.path.exists(output_dir)}")

if not os.path.exists(output_dir):
    try:
        os.makedirs(output_dir, exist_ok=True)
        st.sidebar.success(f"Created output directory: {output_dir}")
    except Exception as e:
        st.sidebar.error(f"Could not create folder: {e}")

# File uploader
uploaded = st.file_uploader(
    "Upload FITS files",
    type=["fit", "fits"],
    accept_multiple_files=True
)

# Debug container for status messages
debug_container = st.container()

# Convert button
if uploaded:
    files_info = ", ".join([f.name for f in uploaded])
    st.write(f"Files ready for conversion: {files_info}")
    
    convert_btn = st.button("▶️ Convert")
    if convert_btn:
        debug_container.info("Convert button clicked!")
        total = len(uploaded)
        progress = st.progress(0)
        status = st.empty()

        # Create a success counter
        success_count = 0
        
        for i, file in enumerate(uploaded, start=1):
            fname = file.name
            status.text(f"Processing **{fname}** ({i}/{total})…")
            try:
                debug_container.info(f"Opening FITS file: {fname}")
                hdul = fits.open(file)
                data = np.nan_to_num(hdul[0].data)
                header = hdul[0].header
                hdul.close()

                # metadata + rendering
                debug_container.info(f"Extracting metadata for: {fname}")
                meta = extract_metadata(header, data)
                name, _ = os.path.splitext(fname)

                png_path = os.path.join(output_dir, f"{name}.png")
                debug_container.info(f"Rendering PNG to: {png_path}")
                render_success = render_png(data, png_path)
                
                if not render_success:
                    debug_container.error(f"Failed to render PNG for {fname}")
                    continue
                    
                json_path = os.path.join(output_dir, f"{name}.json")
                debug_container.info(f"Writing JSON to: {json_path}")
                with open(json_path, "w") as jf:
                    json.dump(meta, jf, indent=2)

                debug_container.success(f"Successfully processed: {fname}")
                status.success(f"✔ {fname} → `{name}.png` + `{name}.json`")
                success_count += 1
            except Exception as e:
                debug_container.error(f"Error processing {fname}: {type(e).__name__}: {e}")
                status.error(f"✖ Error on {fname}: {e}")

            progress.progress(i / total)

        # Only proceed with ZIP if we had successful conversions
        if success_count > 0:
            try:
                debug_container.info("Creating ZIP file...")
                zip_base = os.path.join(output_dir, "converted")
                debug_container.info(f"ZIP base path: {zip_base}")
                
                # Make sure the output directory exists and is accessible
                if not os.path.exists(output_dir):
                    debug_container.error(f"Output directory doesn't exist: {output_dir}")
                    os.makedirs(output_dir, exist_ok=True)
                    debug_container.info(f"Created output directory: {output_dir}")
                
                # Create ZIP file
                shutil.make_archive(base_name=zip_base, format="zip", root_dir=output_dir)
                
                # Verify ZIP was created
                zip_path = zip_base + ".zip"
                debug_container.info(f"ZIP path: {zip_path}")
                debug_container.info(f"ZIP file exists: {os.path.exists(zip_path)}")
                
                if os.path.exists(zip_path):
                    # Add small delay to ensure file is ready
                    time.sleep(0.5)
                    
                    # Create a fallback download option
                    fallback_container = st.container()
                    fallback_container.warning("If the download button doesn't appear below, your ZIP file is still available at:")
                    fallback_container.code(zip_path)
                    
                    # Attempt to open the file for download
                    try:
                        with open(zip_path, "rb") as f:
                            file_size = os.path.getsize(zip_path)
                            debug_container.info(f"ZIP file size: {file_size} bytes")
                            
                            download_button = st.download_button(
                                label="📦 Download all results as ZIP",
                                data=f,
                                file_name="fits_converted_results.zip",
                                mime="application/zip"
                            )
                            debug_container.info("Download button created")
                    except Exception as e:
                        debug_container.error(f"Error creating download button: {type(e).__name__}: {e}")
                        st.error(f"Could not create download button: {e}")
                else:
                    debug_container.error(f"ZIP file not created: {zip_path}")
                    st.error("Failed to create ZIP file for download")
            except Exception as e:
                debug_container.error(f"Error in ZIP creation: {type(e).__name__}: {e}")
                st.error(f"Error creating ZIP file: {e}")
                
            # Create a simple text file download as fallback
            try:
                debug_container.info("Creating fallback text log file")
                log_path = os.path.join(output_dir, "conversion_log.txt")
                with open(log_path, "w") as f:
                    f.write(f"Conversion completed at: {time.ctime()}\n")
                    f.write(f"Files processed: {total}\n")
                    f.write(f"Successfully converted: {success_count}\n")
                
                with open(log_path, "rb") as f:
                    st.download_button(
                        label="📄 Download conversion log",
                        data=f,
                        file_name="conversion_log.txt",
                        mime="text/plain"
                    )
            except Exception as e:
                debug_container.error(f"Error creating fallback download: {type(e).__name__}: {e}")

        st.balloons()
        st.success(f"Conversion complete! {success_count}/{total} files processed successfully.\nFiles written to:\n`{output_dir}`")
else:
    st.info("Upload FITS files to get started.")

# Add a refresh button
if st.button("🔄 Refresh App"):
    st.experimental_rerun()
