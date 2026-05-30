from setuptools import setup, find_packages

setup(
    name="PLred",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "opencv-python",
        # "scikit-learn",
        "astropy",
        "configobj",
    ],
    entry_points={
        "console_scripts": [
            # Calibration tools
            "plred-traces  = PLred.scripts.find_traces_cli:main",
            "plred-wavesol = PLred.scripts.wavesol_cli:main",
            "plred-profile = PLred.scripts.profile_cli:main",
            # Per-step pipeline CLIs
            "plred-sort    = PLred.scripts.sort_cli:main",
            "plred-ingest  = PLred.scripts.ingest_cli:main",
            "plred-roi     = PLred.scripts.roi_cli:main",
            "plred-average = PLred.scripts.average_cli:main",
            "plred-extract = PLred.scripts.extract_cli:main",
            # Master runner
            "plred-run     = PLred.scripts.pipeline_cli:main",
        ],
    },
)
