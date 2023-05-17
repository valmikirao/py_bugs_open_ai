import os
import subprocess

import click
from jinja2 import Template

from py_bugs_open_ai.constants import ROOT_DIR, SHORT_DESCRIPTION, LICENSE, AUTHOR, AUTHOR_EMAIL, CLI_NAME
from py_bugs_open_ai.cli import main as cli_main


def get_config_file_help() -> str:
    config_file_help = ''

    def _format_param_prefix(param_: click.Parameter) -> str:
        param_prefix = param_.name
        if isinstance(param_, click.Option) and param_.is_flag:
            param_prefix += ' (true or false)'
        param_prefix += ': '
        return param_prefix

    ljust_width = max(len(_format_param_prefix(p)) for p in cli_main.params)
    for param in cli_main.params:
        opts_str = ', '.join(param.opts)
        config_file_help += _format_param_prefix(param).ljust(ljust_width) + f" {opts_str}\n"
    return config_file_help


def main():
    rendered = render_readme()
    with open(os.path.join(ROOT_DIR, 'README.md'), 'w') as f:
        f.write(rendered)


def render_readme():
    with open(os.path.join(ROOT_DIR, 'templates', 'README.template.md'), 'r') as f:
        template_str = f.read()
    template = Template(template_str)
    help_message = subprocess.check_output([CLI_NAME, '--help']).decode()
    config_file_help = get_config_file_help()
    rendered = template.render(
        SHORT_DESCRIPTION=SHORT_DESCRIPTION, LICENSE=LICENSE, AUTHOR=AUTHOR, AUTHOR_EMAIL=AUTHOR_EMAIL,
        HELP_MESSAGE=help_message, CONFIG_FILE_HELP=config_file_help
    )
    return rendered


if __name__ == '__main__':
    main()
