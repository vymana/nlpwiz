import setuptools
from nlpwiz.version import Version


setuptools.setup(name='nlpwiz',
                 version=Version('1.0.0').number,
                 description='Python Package Boilerplate',
                 long_description=open('README.md').read().strip(),
                 author='Author Name',
                 author_email='email@domain.com',
                 url='http://path-to-my-packagename',
                 py_modules=['nlpwiz'],
                 install_requires=[],
                 license='MIT License',
                 zip_safe=False,
                 keywords='boilerplate package',
                 classifiers=['Packages', 'nlpwiz'])
