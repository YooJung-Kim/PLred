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
        "astropy"
    ],
)