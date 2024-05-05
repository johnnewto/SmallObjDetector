from setuptools import setup, find_packages

with open("requirements.txt") as f:
    install_requires = f.read().strip().split("\n")
    print(install_requires)

setup(
    name='small_object_detector',
    version='v0.0.1',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    url='https://github.com/johnnewto/SmallObjDetector',
    license='',
    author='john',
    author_email='',
    description='Small Object Detector, ',
    install_requires=install_requires,
)
