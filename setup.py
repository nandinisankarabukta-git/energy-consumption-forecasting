from setuptools import find_packages, setup

setup(
    name='energy-consumption-forecasting',
    version='0.1.0',
    packages=find_packages(),
    description='Machine learning project for predicting electricity consumption across diverse buildings',
    author='Group 6275-1',
    author_email='',
    license='MIT',
    install_requires=[
        'pandas>=1.3.0',
        'numpy>=1.21.0',
        'scikit-learn>=0.24.0',
        'matplotlib>=3.3.0',
        'click',
        'python-dotenv>=0.5.1',
    ],
    python_requires='>=3.7',
)
