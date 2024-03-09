from setuptools import setup, find_packages


VERSION = '1.0'
DESCRIPTION = "A python project to perform gaussian beam optics"
LONG_DESCRIPTION = "Propagate gaussian beams through optics and calculate their properties."

setup(name='GaussianBeam',
      version=VERSION, 
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      author='Lorenzo Versini',
      author_email='lorenzo.versini@physics.ox.ac.uk',
      packages=find_packages(),
      #py_modules=['gaussian_beam', 'cavity'],
      install_requires=['numpy', 'matplotlib', 'scipy'],
        classifiers= [
          "Development Status :: 3 - Alpha",
          "Intended Audience :: Research",
          "Programming Language :: Python :: 3",
          "Operating System :: MacOS :: MacOS X",
          "Operating System :: Microsoft :: Windows",
  ]
      )


