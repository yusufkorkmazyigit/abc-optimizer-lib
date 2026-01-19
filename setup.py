from setuptools import setup, find_packages

setup(
    name="abc_algorithm",
    version="0.1.0",
    description="Kanonik Yapay Arı Kolonisi Algoritması ve Scikit-Learn Wrapper'ı",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scikit-learn"
    ],
    author="Yusuf Korkmazyiğit",
    author_email="yusuf.korkmazyigit@gmail.com"
)