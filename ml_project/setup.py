from setuptools import find_packages, setup

setup(
    name='heart_disease_classifier',
    packages=find_packages(),
    version='0.1.0',
    description='Classifier project based on Heart Disease Dataset from UCI Machine Learning Repository',
    author='Alex Krug',
    install_requires=[
        "numpy==1.21.5",
        "click==8.0.4",
        "scikit-learn==1.0.2",
        "dataclasses==0.6",
        "pyyaml==6.0",
        "marshmallow-dataclass==8.5.8",
        "pandas==1.3.5",
        "catboost==0.26.1",
        "pytest==7.1.2",
    ],
    license='MIT',
)
