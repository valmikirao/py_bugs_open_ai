import os
import re
import subprocess

import click
from jinja2 import Template

from py_bugs_open_ai.constants import ROOT_DIR, SHORT_DESCRIPTION, LICENSE, AUTHOR, AUTHOR_EMAIL, CLI_NAME
from py_bugs_open_ai.cli import main as cli_main


def get_config_file_help() -> str:
    config_file_help = ''

    def _format_param_prefix(param_: click.Parameter) -> str:
        assert param_.name
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


# normally, I would say using an re to parse html or md is evil, but this is such a limited case and I don't
# want to install a whole html parser just for this
ANCHOR_RE = re.compile(r'\b *<a\b +id="(\w+)"/>')
HEADING_RE = re.compile(r'^##\s+(.+)$')


def render_toc(md_str: str) -> str:
    toc = ''
    toc_count = 0
    for line in md_str.split('\n'):
        heading_match = HEADING_RE.search(line)
        if heading_match is not None:
            header_content = heading_match.group(1)
            anchor_match = re.search(ANCHOR_RE, header_content)
            if anchor_match is not None:
                anchor_id = anchor_match.group(1)
                header_text = re.sub(ANCHOR_RE, '', header_content)
                toc_count += 1
                toc += f"{toc_count}. [{header_text}](#{anchor_id})\n"
            else:
                raise AssertionError(f"Error for header {header_content!r}: H2 headers need to have anchors of the"
                                     f" form '<a id=\"AnchorId\"/>'")

    return toc


def render_readme():
    with open(os.path.join(ROOT_DIR, 'templates', 'README.template.md'), 'r') as f:
        template_str = f.read()
    template_0 = Template(template_str)
    help_message = subprocess.check_output([CLI_NAME, '--help']).decode()
    config_file_help = get_config_file_help()
    toc_placeholder = '{{TOC}}'
    rendered_0 = template_0.render(
        SHORT_DESCRIPTION=SHORT_DESCRIPTION, LICENSE=LICENSE, AUTHOR=AUTHOR, AUTHOR_EMAIL=AUTHOR_EMAIL,
        HELP_MESSAGE=help_message, CONFIG_FILE_HELP=config_file_help, TOC=toc_placeholder
    )
    toc = render_toc(rendered_0)
    template_1 = Template(rendered_0)
    rendered_1 = template_1.render(TOC=toc)
    return rendered_1


def main():
    rendered = render_readme()
    readme = os.path.join(ROOT_DIR, 'README.md')
    with open(readme, 'w') as f:
        f.write(rendered)


if __name__ == '__main__':
    main()
