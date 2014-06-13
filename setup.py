from distutils.core import setup

setup(
    name='NeuralNet',
    version='0.1.0',
    author='Wong Liang Ze',
    author_email='liangze.wong@gmail.com',
    packages=['NeuralNet'],
    scripts=['bin/test_fit.py','bin/NeuralNetDemo.py'],    
    license='LICENSE.txt',
    description='Neural Network library.',
    long_description=open('README.md').read(),
    install_requires=[    ],
)