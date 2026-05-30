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
    try:
        from configobj import ConfigObj
        import h5py, numpy as np
        cfg    = ConfigObj(args.config)
        se_cfg = cfg.get('Specextract', {})
        cm_fits   = se_cfg.get('output', 'couplingmap.fits').strip() or 'couplingmap.fits'
        map_h5    = se_cfg.get('input', '').strip() or None
        ext_type  = se_cfg.get('extractor', 'simple_box').strip()
        outdir    = os.path.dirname(os.path.abspath(cm_fits))

        from PLred.scripts._diagnostics import plot_step5_extraction, save_diagnostic

        # Standard coupling map overview (all extractor types)
        fig = plot_step5_extraction(cm_fits, map_h5=map_h5)
        save_diagnostic(fig, outdir, '5_extraction.png')

        # FIRSTPL-specific 3-panel quality check (image / model / residual)
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

                ix, iy = np.unravel_index(np.argmax(nframes), nframes.shape)
                ref_image = avg_plcam[ix, iy]

                from PLred.scripts._diagnostics import plot_firstpl_quality
                fig2 = plot_firstpl_quality(
                    model_file = model_file,
                    ref_image  = ref_image,
                    plcam_roi  = plcam_roi,
                    var_const  = var_const,
                    thresh     = thresh,
                    truncate   = truncate,
                )
                save_diagnostic(fig2, outdir, '5_extraction_quality.png')

    except Exception as e:
        print(f'Warning: could not save step 5 diagnostic: {e}')


if __name__ == '__main__':
    main()
