"""app_rppg.py -- Streamlit console for camera-based two-site PTT.

On file size, which decides the design
--------------------------------------
A 60 s 1080p240 phone clip is roughly 750 MB, and Streamlit's default upload cap is 200 MB.
Raising the cap would still copy the whole file through the browser into a temp directory, which
is slow and pointless when the file is already on this machine. So this app takes a PATH (or a
drag-and-drop of a local file) and reads it in place. Upload is offered only as a fallback for
genuinely small clips, with the limit stated up front rather than discovered as an error.

    streamlit run app_rppg.py
"""
import json
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

st.set_page_config(page_title="camera PTT", layout="wide")
st.title("Camera-based pulse transit time")

tab_rec, tab_an, tab_cmp = st.tabs(["record (webcam)", "analyse video", "compare conditions"])

# --------------------------------------------------------------------- record
with tab_rec:
    st.markdown(
        "Live capture runs in a separate OpenCV window, because Streamlit cannot host the "
        "ROI picker or hit a usable frame rate through the browser. This tab exists to launch "
        "it and to state the limits honestly."
    )
    c1, c2 = st.columns(2)
    secs = c1.number_input("seconds", 15, 300, 60)
    tag = c2.text_input("tag", "rest")
    st.code(f"python rppg_pose.py --seconds {secs} --tag {tag} --stages", language="bash")
    st.warning(
        "The webcam measures ~30 fps, i.e. 33 ms per frame, while arm transit is ~10-20 ms. "
        "Live capture gives reliable heart rate but **not** reliable transit time. For timing, "
        "record 240 fps slow-motion on a phone and use the next tab."
    )

# -------------------------------------------------------------------- analyse
with tab_an:
    st.markdown(
        "Point this at a **240 fps** phone clip. At 240 fps a frame is 4.2 ms, so arm transit "
        "spans several frames instead of sitting under one."
    )
    st.info(
        "Paste a path rather than uploading. A 60 s 1080p240 clip is about 750 MB and "
        "Streamlit's upload cap is 200 MB; reading the file in place also avoids copying it."
    )
    path = st.text_input("video path", "")
    c1, c2, c3 = st.columns(3)
    vtag = c1.text_input("tag ", "phone_rest")
    scale = c2.slider("downscale", 0.25, 1.0, 0.5, 0.25,
                      help="pose detection and patch means are insensitive to resolution; "
                           "downscaling mostly buys speed")
    maxf = c3.number_input("max frames (0 = all)", 0, 100000, 0, 500)

    if path:
        p = Path(path.strip('"').strip())
        if not p.exists():
            st.error(f"not found: {p}")
        else:
            import rppg_video as V
            try:
                info = V.probe(p)
                mb = p.stat().st_size / 1e6
                a, b, c, d = st.columns(4)
                a.metric("resolution", f"{info['w']}×{info['h']}")
                b.metric("frame rate", f"{info['fps']:.0f} fps")
                c.metric("frame quantum", f"{1000/max(info['fps'],1e-9):.1f} ms")
                d.metric("size", f"{mb:.0f} MB")
                if info["fps"] < 120:
                    st.warning(
                        f"{info['fps']:.0f} fps gives a {1000/info['fps']:.0f} ms quantum. "
                        "Arm transit is ~10-20 ms, so timing will be interpolation-limited. "
                        "Heart rate will still be fine."
                    )
                else:
                    st.success(
                        f"{info['fps']:.0f} fps: arm transit spans "
                        f"~{15/(1000/info['fps']):.0f} frames. Good for timing."
                    )

                if st.button("analyse", type="primary"):
                    bar = st.progress(0.0, "decoding and tracking ...")
                    acc, T, fps, dist, seg = V.extract(
                        p, int(maxf), scale, progress=lambda f: bar.progress(min(f, 1.0)))
                    bar.progress(1.0, "analysing ...")
                    res = V.analyse(acc, T, fps, dist, vtag, make_stages=True)
                    bar.empty()

                    a, b, c = st.columns(3)
                    a.metric("consensus HR", f"{res['consensus_hr']:.1f} bpm")
                    b.metric("points accepted", f"{res['n_accepted']}/{res['n_points']}")
                    c.metric("effective rate", f"{res['fps_effective']:.0f} fps")

                    if "pwv_m_s" in res:
                        st.subheader("arrival time vs anatomical distance")
                        dd = np.array(res["distances_cm"]); lg = np.array(res["lags_ms"])
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(figsize=(6, 3.4))
                        ax.scatter(dd, lg, s=22, color="#2f4b7c")
                        xs = np.linspace(dd.min(), dd.max(), 10)
                        ax.plot(xs, np.polyval(np.polyfit(dd, lg, 1), xs), color="#c1543b")
                        ax.set_xlabel("distance from reference (cm)")
                        ax.set_ylabel("arrival lag (ms)")
                        ax.spines[["top", "right"]].set_visible(False)
                        st.pyplot(fig)
                        st.metric("implied PWV", f"{res['pwv_m_s']:.1f} m/s",
                                  delta=f"r = {res['r']:+.2f}")
                        if res.get("pwv_plausible"):
                            st.success(
                                "Inside the 4-12 m/s upper-limb range and monotonic in "
                                "distance. Arrival growing with distance can only be "
                                "propagation, not a fixed processing offset."
                            )
                        else:
                            st.error(
                                "Outside 4-12 m/s or weakly correlated with distance, so the "
                                "lags are artifact-dominated rather than transit."
                            )
                    fp = ROOT / "figures" / f"fig_rppg_stages_{vtag}.png"
                    if fp.exists():
                        st.subheader("signal stages")
                        st.image(str(fp))
            except Exception as e:
                st.error(str(e))

    with st.expander("small clips only: upload instead"):
        up = st.file_uploader("video (<200 MB)", type=["mp4", "mov", "avi", "mkv"])
        if up is not None:
            tmp = DATA / f"_upload_{up.name}"
            tmp.write_bytes(up.getbuffer())
            st.success(f"saved to {tmp} -- paste that path above")

# -------------------------------------------------------------------- compare
with tab_cmp:
    st.markdown(
        "The signed test. Raising the hand lowers local arterial pressure (~0.77 mmHg/cm), "
        "which softens the vessel and slows the pulse, so PTT should **lengthen**. Lowering it "
        "predicts the opposite. A rig that shifts the predicted way for both is measuring "
        "physiology; one that shifts the same way regardless is measuring posture."
    )
    rows = []
    for f in sorted(DATA.glob("rppg_*_*.json")):
        try:
            j = json.loads(f.read_text())
        except Exception:
            continue
        rows.append({"file": f.stem, "tag": j.get("tag", ""),
                     "HR": j.get("consensus_hr") or j.get("hr_bpm"),
                     "fps": j.get("fps_effective") or j.get("fps"),
                     "PWV m/s": j.get("pwv_m_s"),
                     "lag ms": j.get("lag_ms")})
    if rows:
        st.dataframe(rows, use_container_width=True)
    else:
        st.info("no recordings yet")
