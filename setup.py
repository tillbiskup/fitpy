import os
import setuptools

with open(os.path.join(os.path.dirname(__file__), 'VERSION')) as version_file:
    version = version_file.read().strip()

with open(os.path.join(os.path.dirname(__file__), 'README.rst')) as f:
    readme = f.read()

setuptools.setup(
    name='fitpy',
    version=version,
    description='Framework for fitting models to (spectroscopic) data.',
    long_description=readme,
    long_description_content_type='text/x-rst',
    author='Till Biskup',
    author_email='till@till-biskup.de',
    url='https://www.fitpy.de/',
    project_urls={
        'Documentation': 'https://docs.fitpy.de/',
        'Source': 'https://github.com/tillbiskup/fitpy',
    },
    packages=setuptools.find_packages(exclude=('tests', 'docs')),
    keywords=[
        "fitting",
        "spectroscopy",
        "data processing and analysis",
        "reproducible science",
        "reproducible research",
        "good scientific practice",
        "recipe-driven data analysis",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        'Development Status :: 4 - Beta',
        "Topic :: Scientific/Engineering",
    ],
    install_requires=[
        'aspecd>0.6.4',
        'numpy',
        'scipy>=1.7.0',
        'lmfit',
    ],
    extras_require={
        'dev': ['prospector'],
        'docs': ['sphinx', 'sphinx_rtd_theme'],
    },
    python_requires='>=3.7',
)
