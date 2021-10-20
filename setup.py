from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="src",
    version="0.0.1",
    author="Siddhartha Shandilya",
    description="A Healtcare Web Application",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SiddharthaShandilya/Health_Care_AI",
    author_email="siddharthashandilya104@gmail.com",
    packages=["src"],
    python_requires=">=3.7",
    install_requires=[
        "dvc",
        "tensorflow",
        "matplotlib",
        "numpy",
        "pandas",
        "tqdm",
        "PyYAML",
        "boto3" ,
    ]
)