from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension("hog_ext", ["hog.cpp"])
]
setup(
    name="hog_ext",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext}
)