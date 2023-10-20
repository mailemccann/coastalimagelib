from setuptools import setup

setup(
    name="coastalimagelib",
    version="1.1.0",
    author="Maile P. McCann",
    author_email="mailemcc@usc.ed",
    url="https://github.com/mailemccann/coastalimagelib",
    license="MIT",
    packages=['coastalimagelib'],
    description="Package for creating coastal image products.",
    install_requires=['imageio',
                    'matplotlib',
                    'numpy',
                    'opencv-python',
                    'jupyter',
                    'ipykernel',
                    'scikit-image',
                    'scipy',
                    'scikit-image',
                    'scikit-learn',
                    'netCDF4',
                    'multiprocess',
                    'xarray',
                    ]
)
