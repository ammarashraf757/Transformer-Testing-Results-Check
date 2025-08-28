import json
from datetime import datetime
from typing import Dict, Any, List, Tuple
import gradio as gr
import numpy as np
import pandas as pd

APP_TITLE = "220/132 kV, 250/160 MVA Power Transformer - Test Checker"
APP_DESC = (
    "Enter test results (TTR, Winding Resistance, C&DF, Insulation Resistance/PI, and Excitation/‘DES’). "
    "The app checks against typical industry limits. ⚠️ Always confirm with IEC/IEEE or your utility’s acceptance criteria."
)

# Default limits (can be edited in the UI)
DEFAULT_LIMITS: Dict[str, Any] = {
    "ttr_max_pct_dev": 0.5,           # Turns ratio deviation (%)
    "wr_balance_max_pct": 5.0,        # Winding resistance unbalance (%)
    "cdf_max_pf_pct": 0.7,            # C&DF Power factor (%)
    "ir_min_go": 10.0,                # Insulation resistance (GΩ)
    "pi_min": 2.0,                    # Polarization Index (HV)
    "excitation_current_max_pct": 1.5 # Excitation current (% of rated)
}

SEVERITY = {
    "PASS": ("PASS", "✅"),
    "WATCH": ("WATCH", "⚠️"),
    "FAIL": ("FAIL", "❌")
}


def severity_from_bool(ok: bool) -> Tuple[str, str]:
    return SEVERITY["PASS"] if ok else SEVERITY["FAIL"]


def classify_band(value: float, good_max: float, watch_max: float) -> Tuple[str, str]:
    """Classify value into PASS/WATCH/FAIL."""
    if value <= good_max:
        return SEVERITY["PASS"]
    if value <= watch_max:
        return SEVERITY["WATCH"]
    return SEVERITY["FAIL"]


def evaluate(
    ttr_dev_ab, ttr_dev_bc, ttr_dev_ca,
    wr_ohm_a, wr_ohm_b, wr_ohm_c,
    cdf_pf_hv, cdf_pf_lv,
    ir_hv_lv, ir_hv_gnd, ir_lv_gnd, pi_hv,
    exc_current_a, exc_current_b, exc_current_c,
    limits_json
) -> Tuple[str, pd.DataFrame, str]:

    # Parse user limits
    try:
        user_limits = json.loads(limits_json) if limits_json.strip() else {}
    except Exception:
        user_limits = {}
    limits = DEFAULT_LIMITS.copy()
    limits.update({k: v for k, v in user_limits.items() if k in DEFAULT_LIMITS})

    results: List[Dict[str, Any]] = []

    # --- TTR ---
    ttr_devs = np.array([ttr_dev_ab, ttr_dev_bc, ttr_dev_ca], dtype=float)
    ttr_max = float(np.nanmax(np.abs(ttr_devs)))
    ttr_status = classify_band(ttr_max, limits["ttr_max_pct_dev"], limits["ttr_max_pct_dev"] * 2)
    results.append({
        "Test": "Turns Ratio (TTR) deviation (max)",
        "Measured": f"{ttr_max:.3f} %",
        "Limit": f"≤ {limits['ttr_max_pct_dev']} %",
        "Status": f"{ttr_status[1]} {ttr_status[0]}"
    })

    # --- Winding resistance ---
    wr = np.array([wr_ohm_a, wr_ohm_b, wr_ohm_c], dtype=float)
    wr_avg = float(np.nanmean(wr))
    if wr_avg > 0:
        wr_unbalance = (np.nanmax(wr) - np.nanmin(wr)) / wr_avg * 100
    else:
        wr_unbalance = np.nan
    wr_status = classify_band(wr_unbalance, limits["wr_balance_max_pct"], limits["wr_balance_max_pct"] * 1.5)
    results.append({
        "Test": "Winding Resistance unbalance",
        "Measured": f"{wr_unbalance:.3f} %",
        "Limit": f"≤ {limits['wr_balance_max_pct']} %",
        "Status": f"{wr_status[1]} {wr_status[0]}"
    })

    # --- C&DF / PF ---
    cdf_hv_status = classify_band(cdf_pf_hv, limits["cdf_max_pf_pct"], limits["cdf_max_pf_pct"] * 1.5)
    cdf_lv_status = classify_band(cdf_pf_lv, limits["cdf_max_pf_pct"], limits["cdf_max_pf_pct"] * 1.5)
    results.append({
        "Test": "C&DF (HV winding)",
        "Measured": f"{cdf_pf_hv:.3f} %",
        "Limit": f"≤ {limits['cdf_max_pf_pct']} %",
        "Status": f"{cdf_hv_status[1]} {cdf_hv_status[0]}"
    })
    results.append({
        "Test": "C&DF (LV winding)",
        "Measured": f"{cdf_pf_lv:.3f} %",
        "Limit": f"≤ {limits['cdf_max_pf_pct']} %",
        "Status": f"{cdf_lv_status[1]} {cdf_lv_status[0]}"
    })

    # --- Insulation resistance & PI ---
    ir_statuses = [
        ("IR HV–LV", ir_hv_lv, limits["ir_min_go"]),
        ("IR HV–GND", ir_hv_gnd, limits["ir_min_go"]),
        ("IR LV–GND", ir_lv_gnd, limits["ir_min_go"]),
    ]
    for name, val, lim in ir_statuses:
        st = severity_from_bool(val >= lim)
        results.append({"Test": name, "Measured": f"{val:.2f} GΩ", "Limit": f"≥ {lim} GΩ", "Status": f"{st[1]} {st[0]}"})
    pi_status = severity_from_bool(pi_hv >= limits["pi_min"])
    results.append({"Test": "Polarization Index (HV)", "Measured": f"{pi_hv:.2f}", "Limit": f"≥ {limits['pi_min']}", "Status": f"{pi_status[1]} {pi_status[0]}"})

    # --- Excitation / DES ---
    exc = np.array([exc_current_a, exc_current_b, exc_current_c], dtype=float)
    exc_max = float(np.nanmax(exc))
    exc_status = classify_band(exc_max, limits["excitation_current_max_pct"], limits["excitation_current_max_pct"] * 1.5)
    results.append({
        "Test": "Excitation / DES current (max)",
        "Measured": f"{exc_max:.3f} %",
        "Limit": f"≤ {limits['excitation_current_max_pct']} %",
        "Status": f"{exc_status[1]} {exc_status[0]}"
    })

    df = pd.DataFrame(results)

    # --- Overall ---
    if any("❌" in r["Status"] for r in results):
        overall = "❌ FAIL"
    elif any("⚠️" in r["Status"] for r in results):
        overall = "⚠️ WATCH"
    else:
        overall = "✅ PASS"

    report = {
        "metadata": {
            "title": APP_TITLE,
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "transformer": "250/160 MVA, 220/132 kV"
        },
        "limits": limits,
        "results": results,
        "overall": overall
    }
    return overall, df, json.dumps(report, indent=2)


def make_app() -> gr.Blocks:
    with gr.Blocks(title=APP_TITLE) as demo:
        gr.Markdown(f"# {APP_TITLE}\n{APP_DESC}")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Input Test Results")
                ttr_dev_ab = gr.Number(label="TTR deviation AB (%)", value=0.1)
                ttr_dev_bc = gr.Number(label="TTR deviation BC (%)", value=0.1)
                ttr_dev_ca = gr.Number(label="TTR deviation CA (%)", value=0.1)

                wr_ohm_a = gr.Number(label="Winding Resistance A (Ω)", value=0.123)
                wr_ohm_b = gr.Number(label="Winding Resistance B (Ω)", value=0.124)
                wr_ohm_c = gr.Number(label="Winding Resistance C (Ω)", value=0.122)

                cdf_pf_hv = gr.Number(label="C&DF (HV, %)", value=0.3)
                cdf_pf_lv = gr.Number(label="C&DF (LV, %)", value=0.4)

                ir_hv_lv = gr.Number(label="IR HV–LV (GΩ)", value=50)
                ir_hv_gnd = gr.Number(label="IR HV–GND (GΩ)", value=100)
                ir_lv_gnd = gr.Number(label="IR LV–GND (GΩ)", value=100)
                pi_hv = gr.Number(label="Polarization Index HV", value=2.5)

                exc_current_a = gr.Number(label="Excitation current A (%)", value=0.9)
                exc_current_b = gr.Number(label="Excitation current B (%)", value=1.0)
                exc_current_c = gr.Number(label="Excitation current C (%)", value=0.95)

            with gr.Column():
                gr.Markdown("### Limits (JSON editable)")
                limits_json = gr.Code(label="Limits", language="json", value=json.dumps(DEFAULT_LIMITS, indent=2))
                run_btn = gr.Button("Evaluate", variant="primary")

                overall = gr.Textbox(label="Overall Result")
                table = gr.Dataframe(headers=["Test", "Measured", "Limit", "Status"], wrap=True)
                report_json = gr.Code(label="Report (JSON)", language="json")

        run_btn.click(
            fn=evaluate,
            inputs=[ttr_dev_ab, ttr_dev_bc, ttr_dev_ca,
                    wr_ohm_a, wr_ohm_b, wr_ohm_c,
                    cdf_pf_hv, cdf_pf_lv,
                    ir_hv_lv, ir_hv_gnd, ir_lv_gnd, pi_hv,
                    exc_current_a, exc_current_b, exc_current_c,
                    limits_json],
            outputs=[overall, table, report_json]
        )
    return demo


if __name__ == "__main__":
    app = make_app()
    app.launch()
