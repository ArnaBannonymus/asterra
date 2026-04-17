from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from asterra.utils.checks import check_band_names_unique


@dataclass(frozen=True, slots=True)
class BandSchema:
    """A band schema describing how to interpret the last dimension of an EO array.

    Parameters
    ----------
    band_names:
        Canonical band names, in the same order as the array's last dimension.
        Band names must be unique.
    aliases:
        Optional mapping from alternative names to canonical band names.
        Aliases are resolved by :meth:`resolve`.
    metadata:
        Optional per-band metadata keyed by canonical band name.

    Notes
    -----
    Asterra prefers band access by name instead of hard-coded indices.
    """

    band_names: tuple[str, ...]
    aliases: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.band_names) == 0:
            raise ValueError("BandSchema.band_names must be non-empty.")
        if any((not isinstance(n, str) or n.strip() == "") for n in self.band_names):
            raise ValueError("BandSchema.band_names must be non-empty strings.")
        check_band_names_unique(self.band_names)
        unknown_meta = set(self.metadata).difference(self.band_names)
        if unknown_meta:
            raise ValueError(
                "BandSchema.metadata keys must be canonical band names. "
                f"Unknown keys: {sorted(unknown_meta)!r}"
            )

    @classmethod
    def from_names(
        cls,
        band_names: Iterable[str],
        *,
        aliases: Mapping[str, str] | None = None,
        metadata: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> "BandSchema":
        return cls(tuple(band_names), aliases=aliases or {}, metadata=metadata or {})

    def __len__(self) -> int:  # pragma: no cover (trivial)
        return len(self.band_names)

    def resolve(self, name: str) -> str:
        """Resolve a name or alias to a canonical band name."""

        if name in self.band_names:
            return name
        if name in self.aliases:
            canonical = self.aliases[name]
            if canonical not in self.band_names:
                raise ValueError(
                    f"BandSchema alias maps {name!r} -> {canonical!r}, "
                    "but the canonical name is not present in band_names."
                )
            return canonical
        raise KeyError(f"Unknown band name/alias: {name!r}.")

    def index(self, name: str) -> int:
        """Return the integer index of a band by name (after alias resolution)."""

        canonical = self.resolve(name)
        return self.band_names.index(canonical)

    def indices(self, names: Iterable[str]) -> list[int]:
        """Return indices for multiple band names."""

        return [self.index(n) for n in names]

    def select(self, names: Iterable[str]) -> "BandSchema":
        """Return a new :class:`BandSchema` containing only selected bands."""

        canonical = tuple(self.resolve(n) for n in names)
        check_band_names_unique(canonical)
        new_meta = {k: v for (k, v) in self.metadata.items() if k in canonical}
        return BandSchema(band_names=canonical, aliases={}, metadata=new_meta)

    def rename(self, mapping: Mapping[str, str]) -> "BandSchema":
        """Return a new schema with renamed canonical bands."""

        new_names = []
        for n in self.band_names:
            new_names.append(mapping.get(n, n))
        new_names_t = tuple(new_names)
        check_band_names_unique(new_names_t)
        new_meta: dict[str, Mapping[str, Any]] = {}
        for old_name, meta in self.metadata.items():
            new_meta[mapping.get(old_name, old_name)] = meta
        new_aliases: dict[str, str] = {}
        for alias, canonical in self.aliases.items():
            new_aliases[alias] = mapping.get(canonical, canonical)
        return BandSchema(band_names=new_names_t, aliases=new_aliases, metadata=new_meta)

