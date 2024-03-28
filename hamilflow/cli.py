import os
from pathlib import Path

import click

from hamilflow.dataset.config import Dispatcher, read_yaml


@click.group(invoke_without_command=True)
@click.pass_context
def hamilflow(ctx):
    if ctx.invoked_subcommand is None:
        click.echo("Hello {}".format(os.environ.get("USER", "")))
        click.echo(
            "Welcome to HamilFlow. Use `hamilflow --help` to find all the commands."
        )
    else:
        click.echo("Loading Service: %s" % ctx.invoked_subcommand)


@hamilflow.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path), required=True)
def gen(path: Path):
    """Generate dataset based on a config file.

    :param path: where to create the config file.
    """

    click.secho(f"Reading config from: {path}")

    config = read_yaml(path)
    dispatcher = Dispatcher()

    for model_name, model_config in config["models"].items():
        click.secho(model_config)
        model = dispatcher[model_name](**model_config["definition"])
        data = model(**model_config["args"])

        # ... # save data or load into a data container.
