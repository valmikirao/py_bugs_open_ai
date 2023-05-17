# Python Bugs OpenAI

![version](https://img.shields.io/pypi/v/py_bugs_open_ai)
![python versions](https://img.shields.io/pypi/pyversions/py_bugs_open_ai)
![build](https://img.shields.io/github/actions/workflow/status/valmikirao/py_bugs_open_ai/push-workflow.yml?branch=master)

{{SHORT_DESCRIPTION}}

* Free software: {{LICENSE}}

## Installation

```shell
# in local virtual env
$ pip install py-bugs-open-ai

# globally
$ pipx install py-bugs-open-ai
```

## Usage

```shell
# check for bugs in file
$ pybugsai foo.py

# in a repo
$ git ls-files '*.py' | pybugsai --in

# in the diff from master
$ git diff master -- '*.py' | pybugsai --diff-in
```

`pybugsai` makes heavy use of caching and you should make sure to somehow persist the cache if you run it your ci/cd

From the help:

```text
{{HELP_MESSAGE}}
```

The default for any readme can be set in the `[pybugsai]` of the config files (`pybugsai.cfg`, `setup.cfg`, or the
file specified by the `--config` option):

```text
{{CONFIG_FILE_HELP}}
```

## Skipping False Positives

Sometimes, openai is smart enough to interpret comments added to the code

```python
sys.path.join(foo, bar)  # sys in imported earlier (pybugsai)
```

More reliably, you can have it skip certain chunks of code by using their hashes and the `--skip-chunks` option or
the `skip_chunks` argument in the `.cfg` file.  The hashes are reported in the output

```text
foo.py:1-51; 8a49edc09f token count: 390 - ok
foo.py:68-101; 907cf1dc2c token count: 380 - ok
foo.py:103-148; 3156754fe4 token count: 451 - error
foo.py:150-168; 91b78bdac4 token count: 183 - error
foo.py:171-172; 71daa97727 token count: 13 - ok
```

So if you wanted to skip the two above errors, you could do the following:

```text
[pybugsai]
skip_chunks = 3156754fe4,91b78bdac4
```

## TODO

* Allow user to supply examples of bugs and non-bugs
* Be able to give a lot of examples and use embeddings to only include the relevant ones in the request
* More unit tests.  Moooorrrreeee!!!


## Credits

Created by {{AUTHOR}} <{{AUTHOR_EMAIL}}>>

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

* Cookiecutter: https://github.com/audreyr/cookiecutter
* `audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
