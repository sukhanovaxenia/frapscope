"""Analysis configuration: what to load, from where, and how.

The condition set is data, not code. Earlier this module carried a module-level
``CONDITIONS`` list with absolute paths into one machine's home directory, which
made the package unimportable anywhere else and unpublishable as it stood. It now
defines only the schema and a loader; the condition set lives in a file the user
points at, and an example is in ``examples/``.

    frap --config examples/config_ribosomal.py --out results/

A condition names one experimental group, not one replicate. ``source`` is
whatever its loader expects — a directory of ``.lif`` archives, a single archive,
or a CSV — and ``loader_kwargs`` carries any per-condition extraction settings,
such as a region radius that differs because the inclusions do.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Condition:
    """One experimental condition.

    Parameters
    ----------
    display
        Label used in figures, tables and the statistics output. This is the
        name ``--stats-control`` and ``--stats-exclude`` match against, so it
        should be the name the manuscript uses.
    source
        Path handed to the loader. Relative paths resolve against the directory
        containing the config file, not the working directory, so a config and
        its data can be moved together.
    loader
        Key into ``frapscope.loaders.LOADER_REGISTRY``: ``lif``, ``roi_csv`` or
        ``image_digitized``.
    loader_kwargs
        Extraction settings passed through to the loader. Leave a setting out
        rather than restating its default: a value that appears on two
        conditions and not the other three reads as a difference in how they
        were measured, whether or not it changes anything.
    key
        Optional internal identifier, unused by the analysis. See above.
    """

    display: str
    source: str
    loader: str = "lif"
    loader_kwargs: dict[str, Any] = field(default_factory=dict)
    #: Internal identifier, typically the source folder name. Nothing in the
    #: analysis reads it; it is accepted so that configs written against the
    #: earlier schema still load, and it is worth setting where the folder and
    #: the protein disagree — this panel has one such case and it cost weeks.
    key: str | None = None

    def resolve(self, base: Path) -> "Condition":
        """Return a copy with ``source`` resolved against ``base``."""
        p = Path(self.source)
        return Condition(
            display=self.display,
            source=str(p if p.is_absolute() else (base / p).resolve()),
            loader=self.loader,
            loader_kwargs=dict(self.loader_kwargs),
            key=self.key,
        )

    def validate(self) -> None:
        from .loaders import LOADER_REGISTRY

        if not self.display:
            raise ConfigError("every condition needs a non-empty display name")
        if self.loader not in LOADER_REGISTRY:
            raise ConfigError(
                f"condition {self.display!r}: unknown loader {self.loader!r}; "
                f"available: {sorted(LOADER_REGISTRY)}"
            )
        if not Path(self.source).exists():
            raise ConfigError(
                f"condition {self.display!r}: source does not exist: {self.source}"
            )


class ConfigError(ValueError):
    """Raised on a malformed or unresolvable configuration."""


def load_conditions(path: str | Path) -> list[Condition]:
    """Import a config file and return its validated ``CONDITIONS``.

    Sources are resolved against the config file's own directory and every
    condition is checked to exist before any loading begins. Failing here rather
    than inside the loader means a mistyped path is reported once, by name, at
    the start of a run — not as a partial result forty minutes in with one
    condition silently missing from the comparison.
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")

    spec = importlib.util.spec_from_file_location("frap_user_config", path)
    if spec is None or spec.loader is None:
        raise ConfigError(f"cannot import config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    raw = getattr(module, "CONDITIONS", None)
    if not raw:
        raise ConfigError(f"{path} defines no CONDITIONS list")

    conditions = [c.resolve(path.parent) for c in raw]
    missing = []
    for c in conditions:
        try:
            c.validate()
        except ConfigError as exc:
            missing.append(str(exc))
    if missing:
        raise ConfigError("\n  ".join(["invalid configuration:"] + missing))

    seen: set[str] = set()
    for c in conditions:
        if c.display in seen:
            raise ConfigError(f"duplicate condition name {c.display!r}")
        seen.add(c.display)
    return conditions