#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'Click>=8.1.3,<9.0.0',
    'pydantic>=1.10.7,<2.0.0',
    'diskcache>=5.6.1,<6.0.0',
    'openai[embeddings]>=0.27.4,<1.0.0',
    'tiktoken>=0.3.3,<1.0.0',
    'tenacity>=8.2.2,<9.0.0'
]

test_requirements = ['pytest>=3', ]

setup(
    author="Valmiki Rao",
    author_email='valmikirao@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    description="A utility to help use OpenAI to find bugs in large projects or git diffs in python code",
    entry_points={
        'console_scripts': [
            'pybugsai=py_bugs_open_ai.cli:main',
        ],
    },
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='py_bugs_open_ai',
    name='py_bugs_open_ai',
    packages=find_packages(include=['py_bugs_open_ai', 'py_bugs_open_ai.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/valmikirao/py_bugs_open_ai',
    version='0.1.0',
    zip_safe=False,
)
