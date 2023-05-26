#!/usr/bin/env python

"""The setup script."""
import os.path

from setuptools import setup, find_packages

from setup_constants import SHORT_DESCRIPTION, AUTHOR, AUTHOR_EMAIL, CLI_NAME, VERSION

base_dir, _ = os.path.split(__file__)
readme_file = os.path.join(base_dir, 'README.md')
with open(readme_file, 'r') as f:
    readme = f.read()


requirements = [
    'Click>=8.1.3,<9.0.0',
    'pydantic>=1.10.7,<2.0.0',
    'diskcache>=5.6.1,<6.0.0',
    'openai[embeddings]>=0.27.4,<1.0.0',
    'PyYAML>=6.0.0,<7.0.0',
    'tiktoken>=0.3.3,<1.0.0',
    'tenacity>=8.2.2,<9.0.0'
]

test_requirements = ['pytest>=3', ]

setup(
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    description=SHORT_DESCRIPTION,
    entry_points={
        'console_scripts': [
            f"{CLI_NAME}=py_bugs_open_ai.cli:main",
        ],
    },
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='py_bugs_open_ai',
    name='py_bugs_open_ai',
    packages=find_packages(include=['py_bugs_open_ai', 'py_bugs_open_ai.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/valmikirao/py_bugs_open_ai',
    version=VERSION,
    zip_safe=False,
)
