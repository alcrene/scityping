"""A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name="scityping",  # Required
    version="0.6.0",  # Required
    description="Serializable data types for science applications",  # Optional
    long_description=long_description,  # Optional
    long_description_content_type="text/markdown",  # Optional (see note above)
    # # This field corresponds to the "Home-Page" metadata field:
    # # https://packaging.python.org/specifications/core-metadata/#home-page-optional
    # url="https://github.com/pypa/sampleproject",  # Optional
    author="Alexandre René",
    # author_email="author@example.com",  # Optional
    # Classifiers help users find your project by categorizing it.
    #
    # For a list of valid classifiers, see https://pypi.org/classifiers/
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 4 - Beta",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        # "Topic :: Software Development :: Build Tools",
        # Pick your license as you wish
        "License :: OSI Approved :: MIT License",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate you support Python 3. These classifiers are *not*
        # checked by 'pip install'. See instead 'python_requires' below.
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        # "Programming Language :: Python :: 3.9",
        # "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
    ],
    # This field adds keywords for your project which will appear on the
    # project page. What does your project relate to?
    #
    # Note that this is a list of additional keywords, separated
    # by commas, to be used to assist searching for the distribution in a
    # larger catalog.
    keywords="serialization, scientific computing, reproducibility",  # Optional
    # You can just specify package directories manually here if your project is
    # simple. Or you can use find_packages().
    #
    # Alternatively, if you just want to distribute a single Python file, use
    # the `py_modules` argument instead as follows, which will expect a file
    # called `my_module.py` to exist:
    #
    #   py_modules=["my_module"],
    #
    packages=["scityping"],  # Required
    # Specify which Python versions you support. In contrast to the
    # 'Programming Language' classifiers above, 'pip install' will check this
    # and refuse to install the project if the version does not match. See
    # https://packaging.python.org/guides/distributing-packages-using-setuptools/#python-requires
    python_requires=">=3.7, <4",

    install_requires=["pydantic"],

    extras_require={
        # "numpy": ["numpy", "blosc"],
        "dev"  : ["check-manifest"],
        "docs" : ["sphinx", "myst-parser"],
        "test" : ["coverage", "pytest",
                  "numpy", "scipy", "blosc", "torch", "pint", "quantities",
                  ],
    },

    # This field corresponds to the "Project-URL" metadata fields:
    # https://packaging.python.org/specifications/core-metadata/#project-url-multiple-use
    project_urls={
        "Documentation": "https://scityping.readthedocs.io/en/latest/",
        "Source": "https://github.com/alcrene/scityping/",
        "Bug reports": "https://github.com/alcrene/scityping/issues",
    },
)
