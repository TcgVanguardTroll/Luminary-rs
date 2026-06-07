#!/usr/bin/env python3
"""Cross-platform luminary data paths for the Python sidecars.

Mirrors the Rust `dirs` crate's `data_local_dir()` so these scripts resolve the
SAME database/config the `luminary` binary uses, on macOS, Windows, and Linux —
no hardcoded paths. Override with the LUMINARY_DB env var if your DB lives
elsewhere.

  macOS    ~/Library/Application Support/luminary/luminary.db
  Windows  %LOCALAPPDATA%\\luminary\\luminary.db
  Linux    $XDG_DATA_HOME/luminary/luminary.db  (or ~/.local/share/luminary/…)
"""
import os
import sys


def data_local_dir():
    """The OS-standard local-data directory (matches dirs::data_local_dir)."""
    if sys.platform == "win32":
        return os.environ.get("LOCALAPPDATA") or os.path.expanduser(r"~\AppData\Local")
    if sys.platform == "darwin":
        return os.path.expanduser("~/Library/Application Support")
    return os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")


def luminary_dir():
    return os.path.join(data_local_dir(), "luminary")


def db_path():
    """Path to luminary.db (LUMINARY_DB env var wins if set)."""
    return os.environ.get("LUMINARY_DB") or os.path.join(luminary_dir(), "luminary.db")


def config_path():
    return os.path.join(luminary_dir(), "config.json")
