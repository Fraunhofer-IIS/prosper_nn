import setuptools

with open("./prosper_nn/README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

requirements = []
with open("requirements.txt", "r", encoding="utf-8") as f:
    packages = f.read().split("\n")
    for pack in packages:
        requirements.append(pack)

setuptools.setup(
    name="prosper_nn",  # Replace with your own username
    version="0.2.2",
    author="Nico Beck, Julia Schemm",
    author_email="nico.beck@iis.fraunhofer.de",
    description="Package contains, in PyTorch implemented, neural networks with problem specific pre-structuring architectures and utils that help building and understanding models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.6",
)
