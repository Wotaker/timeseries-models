from setuptools import setup, find_packages

setup(
    name='timeseries',
    version='0.1.0',
    packages=find_packages(),
    # TODO uzupełnić requirements
    install_requires=[
        'matplotlib>=3.7',
        'numpy>=1.24',
        'pandas>=2.0',
        'scikit-learn>=1.2',
        'xgboost>=1.7.5',
    ]
)