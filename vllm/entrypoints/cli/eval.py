# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Evaluation commands for an existing OpenAI-compatible vLLM server."""

import argparse
import sys

from vllm.entrypoints.cli.types import CLISubcommand
from vllm.utils.argparse_utils import FlexibleArgumentParser


class EvalSubcommand(CLISubcommand):
    name = "eval"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        from vllm.evals.gsm8k import run_cli

        run_cli(args)

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        parser = subparsers.add_parser("eval", help="Evaluate an existing vLLM server.")
        # Keep evaluation dependencies out of normal serving startup.
        first_positional = next(
            (arg for arg in sys.argv[1:] if not arg.startswith("-")), None
        )
        if first_positional == "eval":
            from vllm.evals.gsm8k import add_cli_args

            evals = parser.add_subparsers(dest="eval_name", required=True)
            add_cli_args(
                evals.add_parser("gsm8k", help="Run the GSM8K accuracy evaluation.")
            )
        return parser


def cmd_init() -> list[CLISubcommand]:
    return [EvalSubcommand()]
