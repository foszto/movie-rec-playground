#!/usr/bin/env python3
import click
from src.cli.preprocess import preprocess
from src.cli.train import train
from src.cli.evaluate import evaluate
from src.cli.recommend import recommend

@click.group()
def cli():
    """MovieLens Recommender System CLI"""
    pass

# Parancsok regisztrálása
cli.add_command(preprocess)
cli.add_command(train)
cli.add_command(evaluate)
cli.add_command(recommend)

if __name__ == '__main__':
    cli()