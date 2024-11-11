from skbuild import setup
from setuptools import find_packages

setup(
    name='tomocupy',
    version=open('VERSION').read().strip(),
    author='Viktor Nikitin',
    package_dir={"": "src"},
    entry_points={'console_scripts':['tomocupy = tomocupy.__main__:main'],},
    packages=find_packages('src'),
    zip_safe=False,
    install_requires=[
    	'cupy',
        'opencv-python',
        'h5py',
        'numexpr',
        'numpy',
        'pywavelets',
        'setuptools',  # for pkg_resources at runtime
        'tifffile',
        'zarr',
    ]
)
