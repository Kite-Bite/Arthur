"""Configuration package: typed schema plus precedence-aware loading."""

from arthur.config.loader import load_config
from arthur.config.schema import Config

__all__ = ["Config", "load_config"]
