"""
plred-extract — Step 5: spectral extraction → coupling map.

Usage
-----
    plred-extract obs.ini
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(
        prog='plred-extract',
        description='Step 5: extract spectra from map.h5 and write couplingmap.fits.',
    )
    parser.add_argument('config', help='Pipeline config .ini file.')
    args = parser.parse_args()

    from PLred.specextract import extract_from_config
    extract_from_config(args.config)

    # ── Diagnostic plots ──────────────────────────────────────────────────────
    import numpy as np
    import h5py
    from configobj import ConfigObj
    from PLred.scripts._diagnostics import (
        plot_step5_extraction, plot_firstpl_quality, save_diagnostic, get_plots_dir,
    )

    cfg      = ConfigObj(args.config)
    se_cfg   = cfg.get('Specextract', {})
    outputs  = cfg.get('Outputs', {})
    cm_fits  = (
        outputs.get('couplingmap_output', '').strip()
        or se_cfg.get('output', '').strip()
        or 'couplingmap.fits'
    )
    map_h5   = (
        outputs.get('average_output', '').strip()
        or se_cfg.get('input', '').strip()
        or None
    )
    ext_type = se_cfg.get('extractor', 'simple_box').strip()
    plots_dir = get_plots_dir(args.config)

    # Coupling map overview (all extractor types)
    save_diagnostic(
        plot_step5_extraction(cm_fits, map_h5=map_h5),
        plots_dir, '5_extraction.png',
    )

    # FIRSTPL 3-panel quality check (image / model / residual)
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
            ref_image = avg_plcam[ix, iy]

            save_diagnostic(
                plot_firstpl_quality(
                    model_file = model_file,
                    ref_image  = ref_image,
                    plcam_roi  = plcam_roi,
                    var_const  = var_const,
                    thresh     = thresh,
                    truncate   = truncate,
                ),
                plots_dir, '5_extraction_quality.png',
            )


if __name__ == '__main__':
    main()
