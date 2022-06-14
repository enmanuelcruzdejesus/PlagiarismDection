from distutils.core import setup

setup(
    name = 'plagiotron',
    version = '0.1.3.1',
    packages = ['src'], 
    install_requires = ['Click', 'nltk', 'termcolor'],
    keywords = ['NLP', 'text', 'text reuse'],
    entry_points='''
    [console_scripts]
    plagiotron = src.plagiotron:cli''',
)
