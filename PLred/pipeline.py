"""
PLred Mode 1 pipeline orchestrator.

Runs the five steps of the spatial response-map / image reconstruction pipeline:

  1  plred-sort    → fastcam.h5   (timestamp matching, PSFcam averaging)
  2  plred-ingest  → alldata.h5   (combined giant H5)
  3  plred-roi     → roi.h5       (ROI viewer cache; interactive pause)
  4  plred-average → map.h5       (spatial sorting / averaging)
  5  plred-extract → couplingmap.fits  (spectral extraction)
"""

import os


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
    from configobj import ConfigObj
    from PLred.scripts._diagnostics import plot_step1, save_diagnostic, get_plots_dir

    script_match_timestamps(configname)

    cfg      = ConfigObj(configname)
    outname  = cfg.get('Output', {}).get('outname', '.').strip() or '.'
    filename = cfg.get('Output', {}).get('filename', 'fastcam').strip() or 'fastcam'
    save_diagnostic(
        plot_step1(os.path.join(outname, filename + '.h5')),
        get_plots_dir(configname), '1_sort.png',
    )


def _step2(configname):
    print("\n=== Step 2: Build giant H5 (plred-ingest) ===")
    from PLred.ingest import ingest_from_config_unified
    from configobj import ConfigObj
    from PLred.scripts._diagnostics import plot_step2, save_diagnostic, get_plots_dir

    ingest_from_config_unified(configname)

    cfg        = ConfigObj(configname)
    alldata_h5 = cfg.get('Ingest', {}).get('output', 'alldata.h5').strip() or 'alldata.h5'
    save_diagnostic(plot_step2(alldata_h5), get_plots_dir(configname), '2_ingest.png')


def _step3(configname, pause=True):
    print("\n=== Step 3: ROI viewer cache (plred-roi) ===")
    from PLred.average import build_ROI_access_from_config
    from configobj import ConfigObj
    from PLred.scripts._diagnostics import plot_step3, save_diagnostic, get_plots_dir

    build_ROI_access_from_config(configname)

    cfg        = ConfigObj(configname)
    alldata_h5 = cfg.get('Ingest', {}).get('output', 'alldata.h5').strip() or 'alldata.h5'
    save_diagnostic(plot_step3(alldata_h5), get_plots_dir(configname), '3_roi.png')

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
    import numpy as np
    import h5py
    from configobj import ConfigObj
    from PLred.specextract import extract_from_config
    from PLred.scripts._diagnostics import (
        plot_step5_extraction, plot_firstpl_quality, save_diagnostic, get_plots_dir,
    )

    extract_from_config(configname)

    cfg       = ConfigObj(configname)
    se_cfg    = cfg.get('Specextract', {})
    cm_fits   = se_cfg.get('output', 'couplingmap.fits').strip() or 'couplingmap.fits'
    map_h5    = se_cfg.get('input', '').strip() or None
    ext_type  = se_cfg.get('extractor', 'simple_box').strip()
    plots_dir = get_plots_dir(configname)

    save_diagnostic(
        plot_step5_extraction(cm_fits, map_h5=map_h5),
        plots_dir, '5_extraction.png',
    )

    if ext_type == 'FIRSTPL' and map_h5:
        model_file = se_cfg.get('model_file', '').strip()
        roi_str    = se_cfg.get('plcam_roi', '').strip()
        var_const  = float(se_cfg.get('var_const', 200) or 200)
        thresh     = float(se_cfg.get('thresh', 0.1) or 0.1)
        truncate   = int(se_cfg.get('truncate', 0) or 0)

        if model_file and roi_str:
            plcam_roi = tuple(int(x) for x in roi_str.split(','))
            with h5py.File(map_h5, 'r') as f:
                avg_plcam = f['avg_PLcam'][:]
                nframes   = f['metadata/nframes'][:]
            ix, iy    = np.unravel_index(np.argmax(nframes), nframes.shape)

            save_diagnostic(
                plot_firstpl_quality(
                    model_file = model_file,
                    ref_image  = avg_plcam[ix, iy],
                    plcam_roi  = plcam_roi,
                    var_const  = var_const,
                    thresh     = thresh,
                    truncate   = truncate,
                ),
                plots_dir, '5_extraction_quality.png',
            )
