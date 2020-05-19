# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================


"""Console script for raytraverse."""
from clasp import click
import clasp.click_ext as clk
import clasp.script_tools as cst
from raytraverse import __version__

@click.group()
@clk.shared_decs(clk.main_decs(__version__))
def main(ctx, config, outconfig, configalias, inputalias):
    """docstring"""
    clk.get_config(ctx, config, outconfig, configalias, inputalias)


@main.command()
@click.argument('arg1')
@clk.shared_decs(clk.command_decs(__version__))
def XXX(ctx, arg1, **kwargs):
    """callbacks for special parsing of command line inputs

Callbacks By type
-----------------

File input
~~~~~~~~~~

file inputs can be given with wildcard expansion (in quotes so that the
callback handles) using glob plus the following:

    * [abc] (one of a, b, or c) 
    * [!abc] (none of a, b or c)
    * '-' (hyphen) collect the stdin into a temporary file (clasp_tmp*)
    * ~ expands user

callback functions:

    * is_file: check if a single path exists (prompts for user input if file
      not found)
    * are_files: recursively calls parse_file_list and prompts on error
    * is_file_iter: use when multiple=True
    * are_files_iter: use when mulitple=True
    * are_files_or_str: tries to parse as files, then tries split_float, then
      split_int, then returns string
    * are_files_or_str_iter: use when mulitple=True

String parsing
~~~~~~~~~~~~~~

    * split_str: split with shlex.split
    * split_str_iter: use when multiple=True
    * color_inp: return alpha string, split on whitespace,
      convert floats and parse tuples on ,

Number parsing
~~~~~~~~~~~~~~

    * tup_int: parses integer tuples from comma/space separated string
    * tup_float: parses float tuples from comma/space separated string
    * split_float: splits list of floats and extends ranges based on : notation
    * split_int: splits list of ints and extends ranges based on : notation
"""
    if kwargs['opts']:
        kwargs['opts'] = False
        clk.echo_args(arg1, **kwargs)
    else:
        try:
            pass
        except click.Abort:
            raise
        except Exception as ex:
            clk.print_except(ex, kwargs['debug'])
    return 'XXX', kwargs, ctx


@main.resultcallback()
@click.pass_context
def printconfig(ctx, opts, **kwargs):
    """callback to save config file"""
    try:
        clk.tmp_clean(opts[2])
    except Exception:
        pass
    if kwargs['outconfig']:
        clk.print_config(ctx, opts, kwargs['outconfig'], kwargs['config'],
                         kwargs['configalias'])


if __name__ == '__main__':
    main()
