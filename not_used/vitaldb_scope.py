"""vitaldb_scope.py -- scope the NATIVE VitalDB clinical data for the mechanism-validation arm.

Why native VitalDB and not PulseDB: PulseDB anonymized VitalDB case IDs into sequential
p000001-style labels, so its subjects cannot be joined to the clinical tables. Recovering the
mapping would mean matching on age/sex/BMI -- a quasi-identifier linkage attack on a
deliberately de-identified release. VitalDB's own API serves waveforms WITH caseid intact
alongside the clinical tables, so the join is native and needs no re-identification.

This script only READS the open clinical tables and reports what is available. It downloads no
waveforms (that is the expensive step, gated on this scoping result).

Question it answers: how many cases have (a) the waveform tracks we need, (b) a documented
vasoactive drug event, and (c) the hemodynamic/lab covariates that make the mechanism claim
falsifiable?
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "vitaldb_scope.json"

API = "https://api.vitaldb.net"
# tracks required for an ECG+PPG arrival-time audit against an arterial reference
NEED = {"art": "SNUADC/ART", "ppg": "SNUADC/PLETH", "ecg": "SNUADC/ECG_II"}
# vasoactive agents that change arterial stiffness / tone -> the causal perturbation
# VitalDB names infusion channels by pump abbreviation (Orchestra/PHEN_RATE), not by full drug
# name -- searching for "PHENYLEPHRINE" matches nothing.
VASO = {"PHEN": "phenylephrine", "NEPI": "norepinephrine", "EPI": "epinephrine",
        "DOPA": "dopamine", "DOBU": "dobutamine", "VASO": "vasopressin",
        "NTG": "nitroglycerin", "NICAR": "nicardipine", "PPF": "propofol",
        "RFTN": "remifentanil"}


def main():
    print("[scope] downloading clinical tables (open access, no DUA) ...", flush=True)
    cases = pd.read_csv(f"{API}/cases")
    trks = pd.read_csv(f"{API}/trks")
    print(f"[scope] cases {cases.shape}, tracks {trks.shape}", flush=True)

    # ---- waveform availability -------------------------------------------------
    have = {}
    for k, tname in NEED.items():
        have[k] = set(trks.loc[trks["tname"] == tname, "caseid"].unique())
        print(f"[scope] {tname:20s} {len(have[k]):5d} cases", flush=True)
    trio = have["art"] & have["ppg"] & have["ecg"]
    print(f"[scope] ART + PLETH + ECG_II together : {len(trio)} cases", flush=True)

    # ---- drug events -----------------------------------------------------------
    tn = trks[trks["caseid"].isin(trio)]
    drug_cases, drug_counts = {}, {}
    for abbr, full in VASO.items():
        m = tn["tname"].str.upper().str.contains(abbr + "_RATE", na=False) | \
            tn["tname"].str.upper().str.contains(abbr + "20_RATE", na=False)
        cs = set(tn.loc[m, "caseid"].unique())
        if cs:
            drug_cases[full] = cs
            drug_counts[full] = len(cs)
    for dv, n in sorted(drug_counts.items(), key=lambda x: -x[1]):
        print(f"[scope]   {dv:16s} {n:5d} cases (with full waveform trio)", flush=True)
    any_vaso = set().union(*drug_cases.values()) if drug_cases else set()
    print(f"[scope] trio + >=1 vasoactive agent   : {len(any_vaso)} cases", flush=True)

    # ---- clinical covariates ---------------------------------------------------
    cc = cases[cases["caseid"].isin(trio)]
    cov = {}
    for c in ["age", "sex", "bmi", "height", "weight", "asa", "preop_htn", "preop_dm",
              "preop_hb", "preop_alb", "preop_cr", "preop_na", "preop_k", "intraop_ebl",
              "intraop_crystalloid", "op_duration", "department", "optype"]:
        if c in cc.columns:
            cov[c] = float(cc[c].notna().mean())
    print("\n[scope] covariate completeness among waveform-trio cases:", flush=True)
    for c, f in sorted(cov.items(), key=lambda x: -x[1]):
        print(f"[scope]   {c:22s} {100*f:5.1f}%", flush=True)

    # hypertensive subgroup is the stiffness contrast that matters most
    if "preop_htn" in cc.columns:
        htn = cc["preop_htn"].fillna(0).astype(float)
        print(f"\n[scope] preop hypertension: {int((htn>0).sum())} of {len(cc)} trio cases "
              f"({100*(htn>0).mean():.0f}%)", flush=True)

    res = {"n_cases_total": int(len(cases)),
           "n_trio": len(trio),
           "n_trio_vasoactive": len(any_vaso),
           "drug_counts": drug_counts,
           "covariate_completeness": cov,
           "trio_caseids": sorted(int(c) for c in trio)[:2000]}
    OUT.write_text(json.dumps(res, indent=2, default=float))
    print(f"\n[done] {OUT}", flush=True)


if __name__ == "__main__":
    main()
