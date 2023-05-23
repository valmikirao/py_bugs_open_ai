# Python Bugs OpenAI

![version](https://img.shields.io/pypi/v/py_bugs_open_ai)
![python versions](https://img.shields.io/pypi/pyversions/py_bugs_open_ai)
![build](https://img.shields.io/github/actions/workflow/status/valmikirao/py_bugs_open_ai/push-workflow.yml?branch=master)

* Free software: GNU General Public License v3

A utility to help use OpenAI to find bugs in large projects or git diffs in python code.  Makes heavy use of caching to save time/money

# Table of Contents

1. [Installation](#Installation)
2. [Usage](#Usage)
3. [System Text](#SystemText)
4. [Skipping False Positives](#Skipping)
5. [Providing Examples](#Examples)
6. [TODO](#TODO)
7. [Credits](#Credits)


## Installation <a id="Installation"/>

```shell
# in local virtual env
$ pip install py-bugs-open-ai

# globally
$ pipx install py-bugs-open-ai
```

## Usage <a id="Usage"/>

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
Usage: pybugsai [OPTIONS] [FILE]...

  Chunks up python files and sends the pieces to open-ai to see if it thinks
  there are any bugs in it

Options:
  -c, --config TEXT               The config file.  Overrides the [pybugsai]
                                  section in pybugsai.cfg and setup.cfg
  --files-from-stdin, --in        Take the list of files from standard in,
                                  such that you could run this script like
                                  `git ls-files -- '*.py' | pybugsai --in`
  --api-key-env-variable TEXT     The environment variable which the openai
                                  api key is stored in  [default:
                                  OPEN_AI_API_KEY]
  --model TEXT                    The openai model used  [default:
                                  gpt-3.5-turbo]
  --embeddings-model TEXT
  --max-chunk-size, --chunk INTEGER
                                  The script tries to break the python down
                                  into chunk sizes smaller than this
                                  [default: 500]
  --abs-max-chunk-size, --abs-chunk INTEGER
                                  Sometimes the script can't break up the code
                                  into chunks smaller than --max-chunk-size.
                                  This is the absolute maximum size of chunk
                                  it will send.  If a chunk is bigger than
                                  this, it will be reported as a warning or as
                                  an error if --strict-chunk-size is set.
                                  Defaults to --max-chunk-size
  --cache-dir, --cache TEXT       The cache directory [~/.pybugsai/cache]
  --refresh-cache
  --die-after INTEGER             After this many errors are found, the
                                  scripts stops running  [default: 3]
  --strict-chunk-size, --strict   If true and there is a chunk that is bigger
                                  than --abs-max-chunk-size, it will be marked
                                  as an error
  --skip-chunks TEXT              The hashes of the chunks to skip.  Can be
                                  added multiple times are be a comma-
                                  delimited list
  --diff-from-stdin, --diff-in    Be able to take `git diff` from the std-in
                                  and then only check the chunks for lines
                                  that are different
  --is-bug-re, --re TEXT          If the response from OpenAI matches this
                                  regular-expression, then it is marked as an
                                  error.  Might be necessary to change this
                                  from the default if you use a customer
                                  --system-content  [default: ^ERROR\b]
  -i, --is-bug-re-ignore-case     Ignore the case when applying the `--is-bug-
                                  re`
  -s, --system-content TEXT       The system content sent to OpenAI
  --examples-file TEXT            File containing example code and responses
                                  to guide openai in finding bugs or non-bugs.
                                  See README for format and more information
                                  [default: ~/.pybugsai/examples.yml]
  --max-tokens-to-send INTEGER    Maximum number of tokens to send to the
                                  OpenAI api, include the examples in the
                                  --examples-file.  pybugsai uses embeddings
                                  to only send the most relevant examples if
                                  it can't send them all without exceeding
                                  this count  [default: 1000]
  --help                          Show this message and exit.

```

The default for any readme can be set in the `[pybugsai]` of the config files (`pybugsai.cfg`, `setup.cfg`, or the
file specified by the `--config` option):

```text
file:                                   file
config:                                 --config, -c
files_from_stdin (true or false):       --files-from-stdin, --in
api_key_env_variable:                   --api-key-env-variable
model:                                  --model
embeddings_model:                       --embeddings-model
max_chunk_size:                         --max-chunk-size, --chunk
abs_max_chunk_size:                     --abs-max-chunk-size, --abs-chunk
cache_dir:                              --cache-dir, --cache
refresh_cache (true or false):          --refresh-cache
die_after:                              --die-after
strict_chunk_size (true or false):      --strict-chunk-size, --strict
skip_chunks:                            --skip-chunks
diff_from_stdin (true or false):        --diff-from-stdin, --diff-in
is_bug_re:                              --is-bug-re, --re
is_bug_re_ignore_case (true or false):  --is-bug-re-ignore-case, -i
system_content:                         --system-content, -s
examples_file:                          --examples-file
max_tokens_to_send:                     --max-tokens-to-send

```

## System Text <a id="SystemText"/>

The `--system-text` argument, `system_text` config variable tells OpenAI what function it should be fulfilling.  Since
the default value was too long to include in the `--help` message, here it is:

```text

```

## Skipping False Positives <a id="Skipping"/>

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

## Providing Examples <a id="Examples"/>

You can provide examples of potential bugs in a file.  By default, the cli looks for this file at
``, but it can also be specified with the `--examples-files` argument.  The file is a Yaml
file with the following format:

```yaml
examples:
  - code: <some code>
    response: <what you wound want OpenAI to respond with for this type of code>
  - <more examples>
```

So, for example:

```yaml
examples:
  - code: os.path.join('dir', 'file')
    response: "OK: Assume that the \"os\" module was imported above"
  - code: my_companys_module.my_companys_function(-1)
    response: "ERROR: my_companys_module.my_companys_function() errors with negative values"
```

If the token count in the query plus the `--system-text` plus the chunk size are greater than `--max-tokens-to-send`,
then the `` will use embeddings to figure out which of the examples are relevant to this particular chunk
and just send those.  Please note that standard billing applies to getting the embeddings.  The embedding results are
cached

If you don't know what embeddings are, this might help explain it:
https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb

## TODO <a id="TODO"/>

* Allow this to use LLM's besides OpenAI
* Add tooling to have some sort of remote cache, so if you run it locally then another contributor or the CI/CD can
  take advantage of the same cache

## Credits <a id="Credits"/>

Created by Valmiki Rao <valmikirao@gmail.com>

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

* Cookiecutter: https://github.com/audreyr/cookiecutter
* `audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage