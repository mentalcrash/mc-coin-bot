"""CLI interface using Typer.

Available subcommands:
    - ingest: Data ingestion pipeline (Bronze -> Silver)
    - backtest: Strategy backtesting and optimization

NOTE: backtest/ingest 모듈은 직접 import하여 사용합니다.
      (python -m src.cli.backtest 실행 시 __init__.py의 eager import가
       RuntimeWarning을 유발하므로, 여기서는 import하지 않습니다.)
"""
