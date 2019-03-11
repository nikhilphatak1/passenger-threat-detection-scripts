
from setuptools import find_packages
from setuptools import setup

'''
REQUIRED_PACKAGES = [
  'tensorflow==1.0.1',
  'tflearn==0.3.2',
  'numpy==1.11.0',
  'scipy==0.17.0',
  'matplotlib==1.5.1'
]
'''


REQUIRED_PACKAGES = [
  'tensorflow==1.0.1',
  'tflearn',
  'numpy',
  'scipy',
  'matplotlib'
]

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    requires=[]
)
