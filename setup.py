import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="calibrary",
    version="0.0.1",
    description="A library of robot arm calibration and optimization procecdures.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maxschommer/calibrary",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'numpy-quaternion',
        'scipy',
        'numba',
        'jax[cpu]'
    ],
    extras_require={
        'debug': [
            'pyvista'
        ]
    },
    python_requires='>=3.6',
)
