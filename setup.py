import setuptools

requirements = [
    'scikit-learn',
    'numpy',
    'pandas',
    'scipy',
    'click',
]


setuptools.setup(
    name="meta_pred",
    version="0.1.0",
    url="https://github.com/Chandrima-04/effet_oppos",
    author="Chandrima Bhattacharya",
    author_email="chb4004@med.cornell.edu",
    description="ML to predict metadata from metagenomes",
    packages=setuptools.find_packages(),
    package_dir={'effet_oppos': 'effet_oppos'},
    entry_points={
        'console_scripts': [
            'effet_oppos=effet_oppos.cli:main',
        ]
    },
    install_requires=requirements,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
