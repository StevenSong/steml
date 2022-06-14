import pathlib
from setuptools import setup, find_packages

here = pathlib.Path(__file__).parent.resolve()
requirements = (here / 'requirements.txt').read_text(encoding='utf-8')
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='steml',
    version='0.0.1',
    author='Steven Song',
    author_email='songs1@uchicago.edu',
    description='Spatial Transcriptomics Enhanced Machine Learning',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/StevenSong/steml',
    python_requires='>=3.8',
    install_requires=requirements,
    packages=find_packages(),
)
