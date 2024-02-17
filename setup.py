""" Theoretical meta-learning project setup. """
from distutils.core import setup

setup(
    name="meta",
    version="0.1dev",
    description="Theoretical meta-learning",
    author="Mikhail Konobeev",
    author_email="konobeev.michael@gmail.com",
    url="https://github.com/MichaelKonobeev/meta/",
    license="MIT",
    packages=["meta"],
    scripts=["meta/scripts/meta"],
    install_requires=[
      "ipython==8.10.0",
      "ipywidgets==7.6.5",
      "jax[cpu]==0.2.27",
      "scipy==1.11.1",
      "matplotlib==3.5.1",
      "numpy==1.22.1",
      "scikit-learn==1.0.2",
      "threadpoolctl==3.0.0",
      "tqdm",
    ],
)
