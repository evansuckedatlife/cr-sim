"""Unified view over a decoded ``csv_logic`` directory.

Modern Clash Royale builds split game logic across two representations:

*   ``*.csv`` -- the legacy wide tables (:mod:`cr_sim.data.csv_loader`).  For any
    entity that has been migrated these rows survive as empty stubs; they still
    carry the definitions for older event/legacy entities.
*   ``*.toml`` -- the current representation.  Each file holds namespaced
    sections such as ``[CHARACTER.Knight]``, ``[BUILDING.Cannon]``,
    ``[PROJECTILE.Arrow]``, ``[AEO....]``, ``[BUFF....]``, ``[ABILITY....]``,
    ``[ACTION....]`` and ``[EXT....]``.

The two never disagree: verified across the 150535029 build, every name present
in both has an empty CSV row, so **TOML wins wherever it is present**.

``EXT`` entries are an inheritance mechanism -- ``Base = "CHARACTER.Foo"`` --
used for Evolutions, champion ability forms and event variants, and the chain
can be several links deep.  :meth:`LogicData.resolve` flattens all of it into a
single attribute dict.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .csv_loader import Table, load_table

__all__ = ["LogicData", "EntityRef", "UnknownEntity", "ENTITY_NAMESPACES"]

# Namespaces that describe a thing that can exist on the battlefield, in the
# order they are searched when a bare name is looked up.
ENTITY_NAMESPACES = ("EXT", "CHARACTER", "BUILDING", "PROJECTILE", "AEO")

# csv file -> the TOML namespace that supersedes it
_CSV_TO_NAMESPACE = {
    "characters": "CHARACTER",
    "buildings": "BUILDING",
    "projectiles": "PROJECTILE",
    "area_effect_objects": "AEO",
    "character_buffs": "BUFF",
    "spells_characters": "SPELL_CHARACTER",
    "spells_buildings": "SPELL_BUILDING",
    "spells_other": "SPELL_OTHER",
    "spells_evolved": "SPELL_EVOLVED",
    "spells_hero_form": "SPELL_HERO",
}

#: TOML-only files whose bare top-level sections belong to a known namespace.
#: Without these their contents would be stranded -- ``spawn_groups`` in
#: particular carries the arena's tower layout.
_TOML_FILE_NAMESPACE = {
    "actions": "ACTION",
    "game_object_filters": "FILTER",
    "spawn_groups": "SPAWN_GROUP",
    "shapes": "SHAPE",
    "damage_types": "DAMAGE_TYPE",
    "target_resolvers": "TARGET_RESOLVER",
    "variables": "VARIABLE",
}

_MAX_BASE_DEPTH = 16

#: Where bare TOML sections go when the file is not an overlay for a CSV table.
_LOOSE_NAMESPACE = "_LOOSE"


def _is_namespace(key: str) -> bool:
    """Namespaces are SHOUTED (``CHARACTER``, ``AEO``, ``SPELL_CHARACTER``);
    entity names are CamelCase (``Archer``, ``DummyKingTower``)."""
    stripped = key.replace("_", "").replace(".", "")
    return bool(stripped) and stripped.isupper()


class UnknownEntity(KeyError):
    """Raised when a name cannot be found in any namespace."""


@dataclass(frozen=True, slots=True)
class EntityRef:
    namespace: str
    name: str

    @classmethod
    def parse(cls, ref: str, default_namespace: str = "CHARACTER") -> "EntityRef":
        if "." in ref:
            ns, _, name = ref.partition(".")
            return cls(ns, name)
        return cls(default_namespace, ref)

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return f"{self.namespace}.{self.name}"


@dataclass(slots=True)
class LogicData:
    """Every logic file from one game build, merged and ready to query."""

    root: Path
    tables: dict[str, Table] = field(default_factory=dict)
    sections: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    _resolved: dict[EntityRef, Mapping[str, Any]] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ load

    @classmethod
    def load(cls, root: str | Path) -> "LogicData":
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"no csv_logic directory at {root}")
        data = cls(root=root)

        for path in sorted(root.glob("*.csv")):
            try:
                data.tables[path.stem] = load_table(path)
            except Exception as exc:  # a few event tables are not standard
                data.tables.pop(path.stem, None)
                data._note_bad_table(path, exc)

        for path in sorted(root.rglob("*.toml")):
            parsed = tomllib.loads(path.read_text(encoding="utf-8"))
            # A TOML file named after a CSV table is an *overlay*: its top-level
            # sections are bare entity names rather than namespaces, and they
            # extend that table.  spells_characters.toml is where Three
            # Musketeers' SummonCharactersList lives, for instance.
            overlay_namespace = _CSV_TO_NAMESPACE.get(path.stem) or _TOML_FILE_NAMESPACE.get(
                path.stem
            )
            for key, entries in parsed.items():
                if not isinstance(entries, dict):
                    continue
                if _is_namespace(key):
                    bucket = data.sections.setdefault(key, {})
                    for name, body in entries.items():
                        if isinstance(body, dict):
                            bucket[name] = body
                elif overlay_namespace is not None:
                    bucket = data.sections.setdefault(overlay_namespace, {})
                    # Merge rather than replace: several files may extend one entity.
                    bucket.setdefault(key, {}).update(entries)
                else:
                    # A bare section in a non-overlay file; keep it addressable
                    # under its own name so nothing is silently lost.
                    data.sections.setdefault(_LOOSE_NAMESPACE, {})[key] = entries
        return data

    def _note_bad_table(self, path: Path, exc: Exception) -> None:
        self.sections.setdefault("_ERRORS", {})[path.stem] = {"error": str(exc)}

    # ----------------------------------------------------------------- query

    def namespace(self, namespace: str) -> Mapping[str, dict[str, Any]]:
        return self.sections.get(namespace, {})

    def names(self, namespace: str) -> tuple[str, ...]:
        """All entity names in ``namespace``, from both TOML and the legacy CSV."""
        found = set(self.sections.get(namespace, {}))
        for stem, ns in _CSV_TO_NAMESPACE.items():
            if ns == namespace and stem in self.tables:
                found.update(r.name for r in self.tables[stem])
        return tuple(sorted(found))

    def _csv_row_for(self, namespace: str, name: str) -> dict[str, Any] | None:
        for stem, ns in _CSV_TO_NAMESPACE.items():
            if ns != namespace or stem not in self.tables:
                continue
            record = self.tables[stem].get(name)
            if record is None:
                continue
            flat = {
                col: values[0]
                for col, values in record.columns.items()
                if values and values[0] is not None
            }
            # Preserve genuine per-level arrays (continuation rows).
            for col in record.columns:
                arr = record.array(col)
                if len(arr) > 1:
                    flat[f"{col}__levels"] = arr
            return flat
        return None

    def _defines(self, ref: EntityRef) -> bool:
        return (
            ref.name in self.sections.get(ref.namespace, {})
            or self._csv_row_for(ref.namespace, ref.name) is not None
        )

    def find(self, name: str) -> EntityRef:
        """Locate a bare name in the entity namespaces."""
        for namespace in ENTITY_NAMESPACES:
            if name in self.sections.get(namespace, {}):
                return EntityRef(namespace, name)
        for namespace in ENTITY_NAMESPACES:
            if self._csv_row_for(namespace, name) is not None:
                return EntityRef(namespace, name)
        raise UnknownEntity(name)

    def locate(self, ref: EntityRef) -> EntityRef:
        """Resolve a reference to the namespace that actually defines it.

        ``Base`` references name the *logical kind* of the entity
        (``Base = "CHARACTER.AngryBarbarian_EV1"``) even when the definition
        lives under ``EXT``, so an exact miss falls back to a name search.
        """
        if self._defines(ref):
            return ref
        for namespace in ENTITY_NAMESPACES:
            candidate = EntityRef(namespace, ref.name)
            if self._defines(candidate):
                return candidate
        raise UnknownEntity(str(ref))

    def resolve(self, ref: "str | EntityRef", _depth: int = 0) -> Mapping[str, Any]:
        """Flatten an entity to a single attribute dict.

        Applies, lowest precedence first: the ``Base`` chain, the legacy CSV row,
        then the TOML section.
        """
        if isinstance(ref, str):
            ref = self.find(ref) if "." not in ref else EntityRef.parse(ref)
        ref = self.locate(ref)
        if ref in self._resolved:
            return self._resolved[ref]
        if _depth > _MAX_BASE_DEPTH:
            raise RecursionError(f"Base chain too deep at {ref}")

        body = self.sections.get(ref.namespace, {}).get(ref.name)
        csv_row = self._csv_row_for(ref.namespace, ref.name)

        merged: dict[str, Any] = {}
        base = (body or {}).get("Base")
        if isinstance(base, str):
            merged.update(self.resolve(EntityRef.parse(base), _depth + 1))
        if csv_row:
            merged.update(csv_row)
        if body:
            merged.update({k: v for k, v in body.items() if k != "Base"})

        merged.setdefault("Name", ref.name)
        merged["__ref__"] = str(ref)
        result: Mapping[str, Any] = merged
        self._resolved[ref] = result
        return result

    def resolve_all(self, namespace: str) -> dict[str, Mapping[str, Any]]:
        out: dict[str, Mapping[str, Any]] = {}
        for name in self.names(namespace):
            try:
                out[name] = self.resolve(EntityRef(namespace, name))
            except (UnknownEntity, RecursionError):
                continue
        return out

    # ---------------------------------------------------------------- globals

    #: value columns of ``globals.csv``, in the order they are consulted
    GLOBAL_VALUE_COLUMNS = ("NumberValue", "FloatValue", "BooleanValue", "TextValue", "StringValue")
    GLOBAL_ARRAY_COLUMNS = ("NumberArray", "NumberArray2", "StringArray")

    def globals_map(self) -> dict[str, Any]:
        """``globals.csv`` flattened to name -> value.

        Each row sets exactly one of the value columns; arrays are exposed under
        a ``<NAME>__<Column>`` key so callers can reach them without a second
        pass over the table.
        """
        table = self.tables.get("globals")
        if table is None:
            return {}
        out: dict[str, Any] = {}
        for record in table:
            for column in self.GLOBAL_VALUE_COLUMNS:
                value = record.scalar(column)
                if value is not None:
                    out[record.name] = value
                    break
            for column in self.GLOBAL_ARRAY_COLUMNS:
                array = record.array(column)
                if array:
                    out[f"{record.name}__{column}"] = array
        return out

    def summary(self) -> dict[str, int]:
        counts = {f"csv:{k}": len(v) for k, v in sorted(self.tables.items())}
        counts.update({f"toml:{k}": len(v) for k, v in sorted(self.sections.items())})
        return counts


def iter_entity_names(data: LogicData, namespaces: Iterable[str] = ENTITY_NAMESPACES):
    for namespace in namespaces:
        for name in data.names(namespace):
            yield EntityRef(namespace, name)
