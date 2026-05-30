"""
PLred Mode 1 pipeline orchestrator.

Runs the five steps of the spatial response-map / image reconstruction pipeline:

  1  plred-sort    → fastcam.h5   (timestamp matching, PSFcam averaging)
  2  plred-ingest  → alldata.h5   (combined giant H5)
  3  plred-roi     → roi.h5       (ROI viewer cache; interactive pause)
  4  plred-average → map.h5       (spatial sorting / averaging)
  5  plred-extract → couplingmap.fits  (spectral extraction)
"""

import sys


ALL_STEPS = [1, 2, 3, 4, 5]


def run_mode1(configname, steps=None):
    """
    Run Mode 1 pipeline steps.

    Parameters
    ----------
    configname : str
        Path to the unified pipeline .ini config.
    steps : list of int or None
        Which steps to run.  None runs all steps (1–5).
        Step 3 (ROI viewer) pauses for user interaction when running in a
        sequence that includes steps after it.
    """
    if steps is None:
        steps = ALL_STEPS

    steps = sorted(set(steps))
    interactive = 3 in steps and any(s > 3 for s in steps)

    for step in steps:
        if step == 1:
            _step1(configname)
        elif step == 2:
            _step2(configname)
        elif step == 3:
            _step3(configname, pause=interactive)
        elif step == 4:
            _step4(configname)
        elif step == 5:
            _step5(configname)
        else:
            raise ValueError(f"Unknown step {step}. Valid steps are 1–5.")


# ---------------------------------------------------------------------------
# Individual step runners
# ---------------------------------------------------------------------------

def _step1(configname):
    print("\n=== Step 1: Timestamp matching (plred-sort) ===")
    from PLred.sort import script_match_timestamps
    script_match_timestamps(configname)


def _step2(configname):
    print("\n=== Step 2: Build giant H5 (plred-ingest) ===")
    from PLred.ingest import ingest_from_config_unified
    ingest_from_config_unified(configname)


def _step3(configname, pause=True):
    print("\n=== Step 3: ROI viewer cache (plred-roi) ===")
    from PLred.average import build_ROI_access_from_config
    build_ROI_access_from_config(configname)
    print("\nROI cache written.")
    if pause:
        print("Open PLred/scripts/h5_viewer.html in a browser to explore the data.")
        print("Fill in [Average] xc, yc, time_min, time_max etc. in your config file.")
        input("\nPress Enter when ready to continue to Step 4 (averaging)...")


def _step4(configname):
    print("\n=== Step 4: Spatial averaging (plred-average) ===")
    from PLred.average import average_to_h5_from_config
    average_to_h5_from_config(configname)


def _step5(configname):
    print("\n=== Step 5: Spectral extraction (plred-extract) ===")
    from PLred.specextract import extract_from_config
    extract_from_config(configname)
