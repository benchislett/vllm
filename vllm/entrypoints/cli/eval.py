# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Evaluation commands for an existing OpenAI-compatible vLLM server."""

import argparse

from vllm.entrypoints.cli.types import CLISubcommand
from vllm.utils.argparse_utils import FlexibleArgumentParser


class EvalSubcommand(CLISubcommand):
    name = "eval"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        from vllm.evals.runner import run_cli

        run_cli(args)

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        from vllm.evals.runner import add_cli_args

        parser = subparsers.add_parser("eval", help="Evaluate an existing vLLM server.")
        add_cli_args(parser)
        return parser


def cmd_init() -> list[CLISubcommand]:
    return [EvalSubcommand()]
