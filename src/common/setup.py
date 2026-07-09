from setuptools import setup, find_packages

setup(
    name="common",
    version="1.1",
    packages=find_packages(),
    install_requires=[
        "pandas==2.3.1",
        "minio==7.2.16",
        "psycopg2-binary==2.9.11"
    ],
)
