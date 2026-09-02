"""The ACTION graph -- this build's scripting language.

Modern Clash Royale builds do not define behaviour in stat columns any more.
They ship **828 ``ACTION`` definitions**: a declarative graph of typed nodes
that spawn things, wait, branch on conditions, and call each other. Anything
interesting added in the last few years lives here rather than in a field --
the reworked Graveyard, the reworked Goblin Hut, champion abilities,
evolutions, and the King Tower's activation.

That is why this module exists instead of a hundred hand-coded special cases.
The graph is data; an interpreter reads it.

Three things had to be worked out from the files before any of it could run.

**Actions are not all in the ``ACTION`` namespace.** ``Graveyard_rework_Group``
lists twelve sub-actions named ``Graveyard_rework_Spawn_Skeleton_1`` through
``_8``, and none of them resolve there -- only a ``_Base`` does. The numbered
ones are ``EXT`` entries extending that base, which is how this format layers a
template plus per-instance overrides. Resolution therefore tries ``ACTION``
first and falls back to ``EXT``; looking only at ``ACTION`` finds the group and
then silently drops every skeleton it was supposed to spawn.

**Positions are expressions, not numbers.** A Graveyard skeleton is placed at::

    XPositionExpression: x + (-2500 * select(x > (map_width / 2), -1, 1))
    YPositionExpression: y - (2500 * team_y_direction(team_index))

Eight of those put a ring of skeletons at 2.5 and 3.5 tiles around the cast
point, mirrored for whichever side of the map and whichever team is casting.
That is the whole reason Graveyard surrounds a tower rather than stacking on
one tile, and it cannot be recovered from any scalar field.

**The expression language is small and closed.** Across every expression-valued
field in the build there are a dozen functions, about fifteen bare names, and
the usual C operators. It is evaluated here by translating ``&&``/``||``/``!``
to their Python spellings, parsing with :mod:`ast`, and walking the tree
against a whitelist of node types -- not with :func:`eval`, which would run
arbitrary code out of a data file.

**An action may be written out in place instead of referenced.** Nineteen
nodes in the build are inline dicts rather than names -- Dark Magic's whole
effect, Goblin Curse's buffs, the Ice Golemite hero's slow ladder -- and every
field that can hold an action can hold either spelling. Anything that accepts
only a name silently drops them, which is exactly how Dark Magic shipped as a
5-elixir spell that did nothing.

Coverage is deliberately partial and deliberately loud. The structural node
types are implemented, along with the one-off classes behind cards a standard
deck can actually play; every class that is *not* is recorded in
:attr:`ActionInterpreter.unsupported` so a card that quietly does nothing shows
up as a name rather than as a mystery. The same counter also records the things
that are not classes at all -- an unreadable condition, a shape this module
cannot measure, a select whose chooser needs a random stream -- under bracketed
keys like ``<condition:...>``.

Four gaps are deliberate rather than pending, because the data does not settle
them:

``ActionTaunt``
    Exists to *clear* a target lock (Goblin Demolisher's transformation runs
    one so its kamikaze form can pick a building). This engine has no taunt and
    no lock, so there is nothing to clear -- left unimplemented rather than
    stubbed, so that adding a taunt mechanic later trips the coverage gate
    instead of inheriting a silent no-op.

``ActionChangeGameObjectData`` with ``NewProjectileData``
    The character form is implemented; the projectile form is not. A unit's
    damage is resolved *from* its projectile at spec-build time, so swapping
    the projectile means rebuilding the damage as well, and the two users (the
    Executioner and Snowball evolutions) give no way to check the result.
    Recorded as ``<changedata:projectile>``.

``ActionSelect`` with ``Condition = "rand(n)"``
    Picks a branch at random. Event content only (the Spell Cauldron, Blackout,
    the gift spawners); this interpreter has no random stream, and always
    taking branch zero would turn the Spell Cauldron into a Lightning
    dispenser. Recorded as ``<select:rand(n)>``.

``ActionGroundToAir``
    The inverse of ``ActionAirToGround`` and only used by a hero form
    (``WizardHero``), whose ability graph is not wired up at all.
"""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from ..data.source import LogicData, UnknownEntity
from .constants import TickClock
from .entity import Entity, EntityKind, Team
from .fixed import SUBTILES_PER_TILE, distance, milli_tiles

__all__ = [
    "ActionContext",
    "ActionInterpreter",
    "ExpressionError",
    "evaluate_expression",
    "matches_filter",
    "MILLI_TILES_PER_TILE",
]

#: Expressions work in milli-tiles, the unit the data files use throughout,
#: while the engine works in subtiles. Converting at the boundary keeps the
#: expression text meaning exactly what it means in the file.
MILLI_TILES_PER_TILE = 1000

#: Arena width in expression units, for ``map_width``. Expressions compare
#: ``x`` against ``map_width / 2`` to decide which way to mirror an offset.
MAP_WIDTH_MILLI_TILES = 18 * MILLI_TILES_PER_TILE


class ExpressionError(ValueError):
    """An expression used something this evaluator does not implement."""


# --------------------------------------------------------------- expressions


def _select(condition: Any, if_true: Any, if_false: Any) -> Any:
    """``select(cond, a, b)`` -- the language's ternary.

    Used almost entirely for mirroring: ``select(x > map_width / 2, -1, 1)``
    flips an offset depending on which half of the board the caster is on, so
    one authored offset covers both lanes.
    """
    return if_true if condition else if_false


def _team_y_direction(team_index: Any) -> int:
    """Which way "forward" points for a team.

    Blue advances up the board and red down it, so an offset authored once is
    correct for both sides. Without this every action written from blue's point
    of view would place its spawns behind a red caster instead of in front.
    """
    return -1 if int(team_index) == int(Team.RED) else 1


#: Everything the expression language may call. A name that is not here is an
#: error rather than a silent zero -- a mistyped or newly added function should
#: surface, not evaluate to nothing.
_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "select": _select,
    "team_y_direction": _team_y_direction,
}

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.Call,
    ast.Name, ast.Constant, ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.FloorDiv,
    ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.IfExp,
)


#: Functions whose arguments are *entity names*, not values to look up.
#: ``has_data(KingTower)`` writes the character's name bare, with no quotes;
#: evaluated as an ordinary name it is simply undefined and the whole
#: expression fails, which would take Vines' and the Ice Golemite hero's
#: buff-size branches down with it.
_NAME_ARGUMENT_FUNCTIONS = frozenset({"has_data"})


def _name_or_value(node: ast.AST, walk: Callable[[ast.AST], Any]) -> Any:
    """An argument to a name-taking function: the bare identifier, or a value.

    ``has_data(-1103717791)`` sits in the same ``||`` chain as
    ``has_data(PrincessTower)``, so this cannot simply stringify everything:
    an identifier is taken literally, anything else is evaluated.
    """
    if isinstance(node, ast.Name):
        return node.id
    return walk(node)


def _to_python(source: str) -> str:
    """Rewrite the C-style operators into Python spellings.

    Only the three that differ. Order matters: ``!=`` must survive the rewrite
    of ``!``, so it is handled by rewriting ``!`` only where a ``=`` does not
    follow.
    """
    out = source.replace("&&", " and ").replace("||", " or ")
    result: list[str] = []
    for index, char in enumerate(out):
        if char == "!" and out[index + 1 : index + 2] != "=":
            result.append(" not ")
        else:
            result.append(char)
    return "".join(result)


def evaluate_expression(source: str, context: Mapping[str, Any]) -> Any:
    """Evaluate one expression from the data files against ``context``.

    Parsed and walked rather than :func:`eval`-ed. These strings come out of a
    game data file, and an interpreter that reached ``eval`` would execute
    whatever a malformed or hostile build put there.
    """
    try:
        # Stripped: the operator rewrite can leave a leading space, and a
        # leading space is an IndentationError to the parser.
        tree = ast.parse(_to_python(source).strip(), mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"cannot parse {source!r}") from exc

    def walk(node: ast.AST) -> Any:
        if not isinstance(node, _ALLOWED_NODES):
            raise ExpressionError(f"{type(node).__name__} not allowed in {source!r}")
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in context:
                return context[node.id]
            if node.id == "true":
                return True
            if node.id == "false":
                return False
            raise ExpressionError(f"unknown name {node.id!r} in {source!r}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ExpressionError(f"computed call in {source!r}")
            name = node.func.id
            if name in _NAME_ARGUMENT_FUNCTIONS:
                args = [_name_or_value(a, walk) for a in node.args]
            else:
                args = [walk(a) for a in node.args]
            if name in _FUNCTIONS:
                return _FUNCTIONS[name](*args)
            if name in context:
                value = context[name]
                return value(*args) if callable(value) else value
            raise ExpressionError(f"unknown function {name!r} in {source!r}")
        if isinstance(node, ast.UnaryOp):
            operand = walk(node.operand)
            if isinstance(node.op, ast.Not):
                return not operand
            if isinstance(node.op, ast.USub):
                return -operand
            return +operand
        if isinstance(node, ast.BoolOp):
            values = [walk(v) for v in node.values]
            return all(values) if isinstance(node.op, ast.And) else any(values)
        if isinstance(node, ast.IfExp):
            return walk(node.body) if walk(node.test) else walk(node.orelse)
        if isinstance(node, ast.Compare):
            left = walk(node.left)
            for op, comparator in zip(node.ops, node.comparators):
                right = walk(comparator)
                if not _compare(op, left, right):
                    return False
                left = right
            return True
        # BinOp
        left, right = walk(node.left), walk(node.right)
        return _arith(node.op, left, right, source)

    return walk(tree)


def _compare(op: ast.cmpop, left: Any, right: Any) -> bool:
    if isinstance(op, ast.Eq):
        return left == right
    if isinstance(op, ast.NotEq):
        return left != right
    if isinstance(op, ast.Lt):
        return left < right
    if isinstance(op, ast.LtE):
        return left <= right
    if isinstance(op, ast.Gt):
        return left > right
    if isinstance(op, ast.GtE):
        return left >= right
    raise ExpressionError(f"unsupported comparison {type(op).__name__}")


def _arith(op: ast.operator, left: Any, right: Any, source: str) -> Any:
    if isinstance(op, ast.Add):
        return left + right
    if isinstance(op, ast.Sub):
        return left - right
    if isinstance(op, ast.Mult):
        return left * right
    if isinstance(op, (ast.Div, ast.FloorDiv)):
        # Integer division throughout. Every consumer of these expressions is a
        # board position, and this engine keeps positions exact by never
        # letting a float into the geometry.
        if right == 0:
            raise ExpressionError(f"division by zero in {source!r}")
        return left // right
    if isinstance(op, ast.Mod):
        return left % right
    raise ExpressionError(f"unsupported operator {type(op).__name__} in {source!r}")


# ------------------------------------------------------------------- context


@dataclass(slots=True)
class ActionContext:
    """Where an action is running: who fired it and from where.

    Position is carried in engine subtiles and converted at the expression
    boundary, so nothing outside this module has to think in milli-tiles.
    """

    team: Team
    x: int
    y: int
    source: Entity | None = None
    #: Named values an action chain has set for later nodes to read.
    variables: dict[str, Any] = field(default_factory=dict)

    def expression_scope(self) -> dict[str, Any]:
        """The names an expression may read."""
        source = self.source
        scope: dict[str, Any] = {
            "x": self.x // 18,
            "y": self.y // 18,
            "map_width": MAP_WIDTH_MILLI_TILES,
            "team_index": int(self.team),
            "is_clone": bool(source is not None and source.is_clone),
            "is_deploying": bool(source is not None and source.is_deploying),
            "is_moving": False,
            "hp": source.hitpoints if source is not None else 0,
            # Both of these ask about the entity the action is *running on*,
            # which for a per-target action is the victim rather than the
            # caster. Vines and the Ice Golemite hero use them to pick which
            # size of snare to draw on what they caught -- ``has_data`` by
            # name, ``get_radius`` by hitbox -- so they are only meaningful
            # with a source, and answer neutrally without one.
            "has_data": self._has_data,
            "get_radius": self._radius_milli_tiles,
        }
        scope.update(self.variables)
        return scope

    def _has_data(self, name: Any) -> bool:
        """``has_data(Knight)`` -- is the entity this ran on that character?

        The argument is a bare name in the file, which the expression walker
        hands over as whatever the surrounding scope made of it; anything that
        is not a string is a numeric character id, and this build carries no
        id-to-name table, so those are answered False rather than guessed.
        """
        source = self.source
        if source is None or source.spec is None or not isinstance(name, str):
            return False
        return source.spec.name == name

    def _radius_milli_tiles(self) -> int:
        """``get_radius()`` -- the hitbox of the entity this ran on, in the
        data's own milli-tiles, which is the unit the comparisons use
        (``get_radius() <= 500`` is "half a tile or smaller")."""
        source = self.source
        return 0 if source is None else source.collision_radius // 18


# ------------------------------------------------------------------ filters


def matches_filter(
    row: Mapping[str, Any] | None, ctx: ActionContext, entity: Entity
) -> bool:
    """Whether one entity passes a ``FILTER`` row, from the caster's point of view.

    ``MatchTeamEnemy`` / ``MatchTeamOwn`` are the load-bearing pair: Vines'
    filter is enemy-only and the Golden Knight's charge finder is too, while
    the Giant Buffer's is own-team. A row that names neither matches both, the
    way ``all_characters_from_both_teams`` does by naming both.

    A ``FILTER`` row is a flat set of flags and only the ones this engine can
    answer are honoured. The rest name states nothing here ever sets --
    ``FilterUnderground``, ``FilterJumping``, ``FilterDragging``,
    ``FilterDashImmune``, ``FilterCloning``, ``FilterHidden``,
    ``FilterIfNoHitpointComponent`` -- and are treated as no-ops rather than as
    rejections, because excluding on a state that is never set would reject
    everything and quietly empty the filter.
    """
    if entity.dead or entity.kind in (EntityKind.PROJECTILE, EntityKind.AREA_EFFECT):
        return False
    if row is None:
        return False
    source = ctx.source
    if row.get("FilterSameObjects") is True and source is not None and entity.id == source.id:
        return False
    if source is not None and entity.id == source.id and row.get("MatchTeamEnemy") is True:
        return False

    own = row.get("MatchTeamOwn") is True
    enemy = row.get("MatchTeamEnemy") is True
    same_team = entity.team is ctx.team
    if own or enemy:
        if same_team and not own:
            return False
        if not same_team and not enemy:
            return False

    is_tower = entity.kind is EntityKind.TOWER
    is_building = entity.kind is EntityKind.BUILDING
    if row.get("FilterTowers") is True and is_tower:
        return False
    if row.get("FilterBuildings") is True and is_building:
        return False
    if row.get("FilterPrincessTowers") is True and is_tower:
        name = entity.spec.name if entity.spec is not None else ""
        if "King" not in name:
            return False
    if row.get("MatchTowers") is True and not is_tower:
        # ``EnemyTowersOnly`` is positive rather than subtractive: it names the
        # only kind it accepts and carries no MatchTypeCharacters at all.
        return False
    if row.get("MatchTypeCharacters") is True and not (
        entity.kind is EntityKind.TROOP or is_building or is_tower
    ):
        return False
    if row.get("FilterFlying") is True and entity.flying:
        return False
    if row.get("FilterClones") is True and entity.is_clone:
        return False
    if row.get("FilterInvisible") is True and not entity.is_acquirable:
        return False
    if row.get("FilterDead") is False:
        pass  # explicitly allowed to include the dead; nothing here does
    if row.get("FilterPushbackIgnore") is True:
        spec = entity.spec
        if spec is not None and spec.ignore_pushback:
            return False
    include = row.get("IncludeCharactersWithData")
    if include:
        if isinstance(include, str):
            include = [include]
        name = entity.spec.name if entity.spec is not None else ""
        if name not in set(include):
            return False
    tags = row.get("FilterTags")
    if isinstance(tags, str) and "UNTARGETABLE" in tags and not entity.is_targetable:
        return False
    return True


# --------------------------------------------------------------- interpreter


#: Node types that exist purely to drive the client: animation, particles,
#: sound, health-bar decoration, ability-button state. Listed explicitly rather
#: than lumped in with the unimplemented ones, because "we chose to ignore
#: this" and "we do not handle this yet" are different facts and only the
#: second is a gap worth reporting.
COSMETIC_CLASSES = frozenset({
    "ActionPlayEffect",
    "ActionPlayAnimationIfHasTarget",
    "ActionRunForcedAnimationOnce",
    "ActionStopForcedAnimation",
    "ActionAnimatorLayer",
    "ActionAddHealthBarPart",
    "ActionOverrideAbilityButtonState",
    "ActionSetAnimationModifier",
    "ActionEnabbleHPBarConditionForDuration",
    "ActionSetIndicatorOnTarget",
    "ActionVisualActionGroup",
    "ActionHide",
    "ActionSetAttackSequenceIndex",
    "ActionChaosS2BadgeTracker",
    "ActionTimerQuest",
})


@dataclass(slots=True)
class _Pending:
    due_tick: int
    action: "str | Mapping[str, Any]"
    context: ActionContext


class ActionInterpreter:
    """Runs ACTION graphs on behalf of a battle.

    Delays are real and load-bearing -- a Graveyard's twelve skeletons are
    authored as twelve sub-actions at 2200ms through 8200ms -- so this keeps a
    queue rather than executing a whole graph at once, and the battle drains
    whatever is due each tick.
    """

    __slots__ = (
        "data", "clock", "_spawn", "_clone", "_place_area", "_count_units",
        "_apply_buff_action", "_pending", "unsupported", "_cache",
        "_nearby", "_set_grounded", "_change_data", "_deal_damage",
        "_arm_counter", "_fire_projectile", "_filter_cache", "_shape_cache",
    )

    def __init__(
        self,
        data: LogicData,
        clock: TickClock,
        spawn: Callable[..., list[Entity]],
        clone: Callable[[Entity], Entity | None] | None = None,
        place_area: Callable[..., Any] | None = None,
        count_units: Callable[[Team, set[str]], int] | None = None,
        apply_buff: Callable[..., None] | None = None,
        *,
        nearby: Callable[[int, int, int], list[Entity]] | None = None,
        set_grounded: Callable[[Entity, int], None] | None = None,
        change_data: Callable[[Entity, str, bool], bool] | None = None,
        deal_damage: Callable[[int, Entity, int], None] | None = None,
        arm_counter: Callable[[Entity, Mapping[str, Any], "ActionContext"], None] | None = None,
        fire_projectile: Callable[[Entity, str, int, int], None] | None = None,
    ) -> None:
        self.data = data
        self.clock = clock
        #: Injected rather than reached for, so this module never imports the
        #: battle and the two can be tested apart.
        self._spawn = spawn
        self._clone = clone or (lambda entity: None)
        self._place_area = place_area or (lambda *a, **k: None)
        self._count_units = count_units or (lambda team, names: 0)
        self._apply_buff_action = apply_buff or (lambda ctx, name, row: None)
        #: Everything within a radius of a point, unfiltered. The *filtering*
        #: stays here rather than in the battle, because a filter is a
        #: ``FILTER`` row and only this module resolves data.
        self._nearby = nearby or (lambda x, y, radius: [])
        self._set_grounded = set_grounded or (lambda entity, ticks: None)
        self._change_data = change_data or (lambda entity, name, reset: False)
        self._deal_damage = deal_damage or (lambda source_id, target, amount: None)
        self._arm_counter = arm_counter or (lambda entity, row, ctx: None)
        self._fire_projectile = fire_projectile or (lambda source, name, x, y: None)
        self._filter_cache: dict[str, Mapping[str, Any] | None] = {}
        self._shape_cache: dict[str, int | None] = {}
        self._pending: list[_Pending] = []
        #: ClassTypes seen but not implemented, with how often. The engine's
        #: gaps should be enumerable rather than discovered one card at a time.
        self.unsupported: Counter[str] = Counter()
        self._cache: dict[str, Mapping[str, Any] | None] = {}

    # -- resolution --------------------------------------------------------

    def resolve(self, name: str) -> Mapping[str, Any] | None:
        """Find an action by name, in ``ACTION`` or failing that ``EXT``.

        The fallback is not a nicety. Graveyard's numbered spawn nodes exist
        only as ``EXT`` entries extending a shared base, so an ACTION-only
        lookup finds the group that schedules twelve skeletons and then cannot
        find a single one of them.
        """
        if name in self._cache:
            return self._cache[name]
        found: Mapping[str, Any] | None = None
        for namespace in ("ACTION", "EXT"):
            try:
                found = self.data.resolve(f"{namespace}.{name}")
                break
            except (UnknownEntity, KeyError):
                continue
        self._cache[name] = found
        return found

    # -- scheduling --------------------------------------------------------

    def schedule(
        self,
        name: "str | Mapping[str, Any]",
        context: ActionContext,
        tick: int,
        delay_ms: Any = 0,
    ) -> None:
        self._pending.append(
            _Pending(tick + self.clock.ticks(delay_ms), name, context)
        )

    def start(
        self, name: "str | Mapping[str, Any]", context: ActionContext, tick: int
    ) -> None:
        """Begin an action graph, honouring the entry node's own start delay.

        ``ActionDelay`` on the entry node is a real part of the behaviour: the
        reworked Goblin Hut waits 1000ms before its first Spear Goblin.
        Running the graph immediately makes every such card produce its first
        output on the tick it lands.

        Takes an inline row as well as a name, for the same reason
        :meth:`run` does: two area effects in this build write their entry node
        out in place instead of referencing one.
        """
        row = name if isinstance(name, Mapping) else self.resolve(name)
        delay = row.get("ActionDelay", 0) if row is not None else 0
        if isinstance(delay, int) and delay > 0:
            self.schedule(name, context, tick, delay)
        else:
            self.run(name, context, tick)

    def run(self, name: "str | Mapping[str, Any]", context: ActionContext, tick: int) -> None:
        """Execute one action now, scheduling whatever it defers.

        ``name`` may be a name to look up or the action itself. Nineteen
        actions in this build are written inline rather than referenced --
        Goblin Curse's buffs and Dark Magic's whole effect among them -- and a
        reader that only accepted names dropped every one of them silently.
        """
        if isinstance(name, Mapping):
            row: Mapping[str, Any] | None = name
        else:
            row = self.resolve(name)
        if row is None:
            self.unsupported[f"<missing:{name}>"] += 1
            return
        class_type = str(row.get("ClassType", ""))
        if class_type in COSMETIC_CLASSES:
            return
        handler = _HANDLERS.get(class_type)
        if handler is None:
            self.unsupported[class_type] += 1
            return
        if not self._passes(row, context, class_type):
            return
        handler(self, row, context, tick)

    def drain(self, tick: int) -> None:
        """Run everything due on this tick. Called once per tick by the battle."""
        if not self._pending:
            return
        due = [p for p in self._pending if p.due_tick <= tick]
        if not due:
            return
        self._pending = [p for p in self._pending if p.due_tick > tick]
        for item in due:
            source = item.context.source
            if source is not None and source.dead:
                # A dead instigator's queue is abandoned. A hut destroyed at 12
                # seconds must stop producing, and a self-rescheduling spawner
                # whose source is gone would otherwise run for the rest of the
                # match with nothing on the board to justify it.
                continue
            self.run(item.action, item.context, tick)

    def _passes(
        self, row: Mapping[str, Any], context: ActionContext, class_type: str = ""
    ) -> bool:
        """Gate an action on its own condition, where it carries one.

        ``Condition`` means two different things depending on the node.
        Everywhere else it is a gate, but on ``ActionSelect`` it is the
        *chooser* -- ``Condition = "rand(6)"`` on the Spell Cauldron picks
        which of six sub-actions to run, and reading it as a gate would let a
        card through on an unevaluable expression while still running nothing.
        So it is left to :func:`_handle_select`.
        """
        keys = (
            ("ExecuteIfTrue",)
            if class_type == "ActionSelect"
            else ("ExecuteIfTrue", "Condition")
        )
        for key in keys:
            expression = row.get(key)
            if not isinstance(expression, str) or not expression.strip():
                continue
            try:
                if not evaluate_expression(expression, context.expression_scope()):
                    return False
            except ExpressionError:
                # An unevaluable condition is recorded and treated as open. A
                # gate nobody can read should not silently disable a card.
                self.unsupported[f"<condition:{expression}>"] += 1
        return True

    # -- object queries ----------------------------------------------------

    def filter_row(self, name: Any) -> Mapping[str, Any] | None:
        """Resolve a ``FILTER`` row by name, cached."""
        if not isinstance(name, str) or not name:
            return None
        if name not in self._filter_cache:
            try:
                self._filter_cache[name] = self.data.resolve(f"FILTER.{name}")
            except (UnknownEntity, KeyError):
                self._filter_cache[name] = None
                self.unsupported[f"<filter:{name}>"] += 1
        return self._filter_cache[name]

    def shape_radius(self, name: Any) -> int | None:
        """A ``SHAPE`` row's reach in engine subtiles, or ``None``.

        Every shape an action actually points at in this build is a
        ``Circle``; the two that are not (the Baby Dragon evolution's
        ``Rectangle`` wind gust, the Mega Minion hero's ``Global``) belong to
        nodes this interpreter does not run, and are recorded rather than
        approximated -- squashing a rectangle into a circle would change which
        units a gust catches.
        """
        if not isinstance(name, str) or not name:
            return None
        if name in self._shape_cache:
            return self._shape_cache[name]
        radius: int | None = None
        try:
            row = self.data.resolve(f"SHAPE.{name}")
        except (UnknownEntity, KeyError):
            row = None
        if row is None:
            self.unsupported[f"<shape:{name}>"] += 1
        elif str(row.get("ClassType", "")) != "Circle":
            self.unsupported[f"<shape:{row.get('ClassType')}>"] += 1
        else:
            value = row.get("Radius")
            radius = milli_tiles(value) if isinstance(value, int) else None
        self._shape_cache[name] = radius
        return radius

    def objects_in_radius(
        self,
        ctx: ActionContext,
        x: int,
        y: int,
        radius: int,
        filter_name: Any,
    ) -> list[Entity]:
        """Everything inside ``radius`` of a point that a ``FILTER`` accepts.

        Radius is in engine subtiles and measured centre-to-centre plus the
        candidate's own hitbox, the same way :func:`cr_sim.engine.targeting.
        gap_between` measures reach -- a Giant standing with its edge inside a
        Vines circle is caught by it.
        """
        row = self.filter_row(filter_name)
        found = [
            entity
            for entity in self._nearby(x, y, radius)
            if distance(x, y, entity.x, entity.y) <= radius + entity.collision_radius
            and matches_filter(row, ctx, entity)
        ]
        found.sort(key=lambda e: e.id)
        return found

    # -- position ----------------------------------------------------------

    def position_for(
        self, row: Mapping[str, Any], context: ActionContext
    ) -> tuple[int, int]:
        """Where a spawning action puts its unit, in engine subtiles."""
        x, y = context.x, context.y
        scope = context.expression_scope()
        for key, axis in (("XPositionExpression", 0), ("YPositionExpression", 1)):
            expression = row.get(key)
            if not isinstance(expression, str) or not expression.strip():
                continue
            try:
                value = int(evaluate_expression(expression, scope)) * 18
            except (ExpressionError, TypeError, ValueError):
                self.unsupported[f"<position:{expression}>"] += 1
                continue
            if axis == 0:
                x = value
            else:
                y = value
        # Fixed relative offsets, for the nodes that use them instead of an
        # expression. Two field names carry the identical shape -- a
        # *whole-tile* offset from the cast point, mirrored on the Y axis by
        # team facing -- and every row in the build uses one or the other,
        # never both: Furnace's forward-spawn is ``MirroredY``, Goblin
        # Drill's four corner spawns are ``RelativeX``/``RelativeY``.
        # "Mirrored" does not mean the engine mirrors it at runtime; paired
        # actions like ``SpawnGoblin1``/``_2`` already carry opposite
        # literal signs, authored once per side.
        #
        # Whole tiles, not milli-tiles: Furnace's ``MirroredY`` is 3, and the
        # card visibly launches Fire Spirits a few tiles ahead of itself, not
        # a hundredth of a tile away. That is a different unit from the
        # ``XPositionExpression``/``YPositionExpression`` pair above, which
        # *is* milli-tiles because the expression scope's ``x``/``y`` are.
        for x_key, y_key in (("RelativeX", "RelativeY"), ("MirroredX", "MirroredY")):
            rel_x, rel_y = row.get(x_key), row.get(y_key)
            if isinstance(rel_x, int):
                x += rel_x * SUBTILES_PER_TILE
            if isinstance(rel_y, int):
                y += rel_y * SUBTILES_PER_TILE * _team_y_direction(int(context.team))
        return x, y


# -------------------------------------------------------------------- nodes

Handler = Callable[[ActionInterpreter, Mapping[str, Any], ActionContext, int], None]


def _handle_group(
    interp: ActionInterpreter, row: Mapping[str, Any], ctx: ActionContext, tick: int
) -> None:
    """``ActionGroup``: sub-actions, each with its own delay.

    ``SubActions`` and ``SubActionsDelay`` are parallel arrays. Graveyard's
    group is twelve spawns at 2200 through 8200ms, which *is* the card's
    trickle -- there is no rate field anywhere else that produces it.
    """
    subs = row.get("SubActions") or ()
    delays = row.get("SubActionsDelay") or ()
    if isinstance(subs, str):
        subs = [subs]
    for index, sub in enumerate(subs):
        if not isinstance(sub, (str, Mapping)) or (isinstance(sub, str) and not sub):
            continue
        delay = delays[index] if index < len(delays) else 0
        interp.schedule(sub, ctx, tick, delay)


def _handle_spawn(
    interp: ActionInterpreter, row: Mapping[str, Any], ctx: ActionContext, tick: int
) -> None:
    """``ActionSpawn`` / ``ActionSpawnToLocation``: put a character on the board."""
    what = row.get("SpawnData")
    spawn_type = str(row.get("SpawnType", "CharacterType"))
    if spawn_type == "BuffType":
        # Goblin Curse works this way: the cloud hits nothing itself, and its
        # per-hit action puts two buffs on whatever it touched -- the curse
        # that spawns a Goblin for the caster when the victim dies, and the
        # damage-over-time. Applied to the entity the action ran on.
        #
        # The buff is usually a name in the ``BUFF`` namespace, but Dark
        # Magic's three damage tiers are written out inside the action and
        # exist nowhere else: ``DarkMagicAOE_Damage_lv1/2/3`` are not in
        # ``BUFF`` at all, so reducing the row to its ``Name`` and looking
        # that up finds nothing and the spell lands for zero. The whole
        # mapping is handed down instead, and the buff layer builds from it.
        if isinstance(what, Mapping) or (isinstance(what, str) and what):
            interp._apply_buff_action(ctx, what, row)
        return
    if isinstance(what, Mapping):
        what = str(what.get("Name") or "")
    if not isinstance(what, str) or not what:
        return
    if spawn_type in ("AreaEffectType", "AreaEffectObject"):
        # An action can place a cloud instead of a unit. Routed to the area
        # effect path rather than spawned as a troop, which would put a
        # 1-hitpoint object on the board for the enemy to walk into.
        x, y = interp.position_for(row, ctx)
        interp._place_area(ctx.team, what, x, y, ctx.source)
        return
    if spawn_type not in ("CharacterType", "", "Character"):
        interp.unsupported[f"<spawntype:{spawn_type}>"] += 1
        return
    x, y = interp.position_for(row, ctx)
    count = row.get("SpawnNumber")
    interp._spawn(
        team=ctx.team,
        character=what,
        count=count if isinstance(count, int) and count > 0 else 1,
        x=x,
        y=y,
        deploy_ticks=interp.clock.ticks(row.get("DeployTime")),
    )


def _handle_next(
    interp: ActionInterpreter, row: Mapping[str, Any], ctx: ActionContext, tick: int
) -> None:
    """Nodes whose only job is to hand off to another after a delay.

    The successor may be written inline rather than referenced -- the Inferno
    Dragon evolution's stage picker and Dark Magic's per-tier spawns both do
    that -- so a Mapping is as valid a target here as a name.
    """
    for key in ("NextAction", "ActionToExecute"):
        target = row.get(key)
        if isinstance(target, Mapping) or (isinstance(target, str) and target):
            interp.schedule(target, ctx, tick, row.get("ActionDelay", 0))


def _handle_select(
    interp: ActionInterpreter, row: Mapping[str, Any], ctx: ActionContext, tick: int
) -> None:
    """``ActionSelect``: run **one** of ``SubActions``, chosen by condition.

    Every one of the twenty-two ``ActionSelect`` nodes in this build uses
    ``SubActions``; not one carries the ``NextAction`` this was previously
    routed through, so the node did nothing anywhere. That silently removed
    Vines' snare (its buff is chosen here), the Hunter evolution's net, the
    P.E.K.K.A evolution's heal-on-kill, the Inferno Dragon evolution's
    attack-stage ladder and the Mega Knight evolution's uppercut cadence.

    ``PerActionConditions`` is one shorter than ``SubActions`` wherever a
    default exists: the first condition that holds picks its sub-action, and
    falling off the end picks the last entry. Vines is 6 conditions to 7
    branches (King Tower, Princess Tower, big troop, medium troop, small
    hitbox, medium hitbox, else); the Mini P.E.K.K.A hero form is 4 to 4 and
    genuinely runs nothing when none match.

    The other spelling -- ``Condition = "rand(6)"`` -- picks an index at
    random. It appears only on event content (Spell Cauldron, Blackout, the
    gift spawners), and this engine gives the interpreter no random stream, so
    it is recorded as a gap rather than resolved to a fixed branch: always
    picking index zero would make the Spell Cauldron a Lightning dispenser.
    """
    subs = row.get("SubActions") or ()
    if isinstance(subs, (str, Mapping)):
        subs = [subs]
    if not subs:
        return

    conditions = row.get("PerActionConditions")
    if not conditions:
        chooser = row.get("Condition")
        if isinstance(chooser, str) and chooser.strip():
            interp.unsupported[f"<select:{chooser}>"] += 1
        return

    if isinstance(conditions, str):
        conditions = [conditions]
    scope = ctx.expression_scope()
    chosen = len(conditions)  # the default branch, where one exists
    for index, expression in enumerate(conditions):
        if not isinstance(expression, str) or not expression.strip():
            continue
        try:
            if evaluate_expression(expression, scope):
                chosen = index
                break
        except ExpressionError:
            interp.unsupported[f"<condition:{expression}>"] += 1
    if chosen >= len(subs):
        return
    interp.schedule(subs[chosen], ctx, tick, row.get("ActionDelay", 0))
    # A select may still chain onward -- the Inferno Dragon evolution follows
    # its stage pick with a tag-setting node.
    _handle_next(interp, row, ctx, tick)


def _handle_set_variable(
    interp: ActionInterpreter, row: Mapping[str, Any], ctx: ActionContext, tick: int
) -> None:
    name = row.get("Variable")
    if isinstance(name, str) and name:
        ctx.variables[name] = row.get("Value", 0)
    _handle_next(interp, row, ctx, tick)


def _handle_interval(
    interp: ActionInterpreter, row: Mapping[str, Any], ctx: ActionContext, tick: int
) -> None:
    """``ActionInterval``: repeat a child action on a period.

    Bounded by ``ActionDuration`` where present. An unbounded repeat would
    schedule itself forever and outlive the match, so a node with no duration
    fires once and is recorded -- silently looping is the worse failure.
    """
    duration = row.get("ActionDuration")
    interval = row.get("Interval") or row.get("ActionDelay")
    target = row.get("ActionToExecute") or row.get("NextAction")
    if not isinstance(target, str) or not target:
        return
    if not isinstance(interval, int) or interval <= 0:
        interp.schedule(target, ctx, tick, 0)
        interp.unsupported["<interval:no-period>"] += 1
        return
    if not isinstance(duration, int) or duration <= 0:
        # No duration means "for as long as I exist". Repeating one step at a
        # time lets the dead-instigator guard end it, which is what a Furnace
        # destroyed mid-cycle needs; unrolling a fixed count instead would
        # either cut it short or outlive the building.
        interp.schedule(target, ctx, tick, 0)
        interp.schedule(row["Name"], ctx, tick, interval)
        return
    elapsed = 0
    while elapsed < duration:
        interp.schedule(target, ctx, tick, elapsed)
        elapsed += interval


def _handle_hut_life_state(
    interp: ActionInterpreter, row: Mapping[str, Any], ctx: ActionContext, tick: int
) -> None:
    """``ActionGoblinHutLifeState``: the reworked huts' whole spawn cycle.

    Goblin Hut and its kin were rebuilt around this node, and their stat
    columns emptied out -- ``SpawnCharacter`` on ``BUILDING.GoblinHut_Rework``
    is the empty string, so the ordinary spawner path finds nothing and the
    building sits inert for its whole 30-second life.

    It reschedules itself rather than unrolling the whole cycle up front,
    because the hut can die at any point and the queue is abandoned when its
    instigator does. Unrolling would keep producing Spear Goblins out of a
    building that is no longer there.
    """
    what = row.get("SpawnData")
    interval = row.get("SpawnInterval")
    if not isinstance(what, str) or not what:
        return
    source = ctx.source
    if source is None or source.dead:
        return

    count = row.get("SpawnNumber")
    offset = row.get("SpawnOffset")
    y = ctx.y
    if isinstance(offset, int) and offset:
        # Spawned in front of the hut, on the side it is fighting toward.
        y += offset * 18 * _team_y_direction(int(ctx.team))
    interp._spawn(
        team=ctx.team,
        character=what,
        count=count if isinstance(count, int) and count > 0 else 1,
        x=ctx.x,
        y=y,
    )
    if isinstance(interval, int) and interval > 0:
        interp.schedule(row["Name"], ctx, tick, interval)


def _handle_run_if_exists(
    interp: ActionInterpreter, row: Mapping[str, Any], ctx: ActionContext, tick: int
) -> None:
    """``ActionRunIfGameObjectExists``: branch on what is on the board.

    Counts friendly units whose name is in ``MatchName`` and runs one action if
    there are at least ``NumMatchesNeeded`` of them, another if not. The Boss
    Bandit uses it to greet her gang, so the branch not taken matters as much
    as the branch taken -- falling through silently would leave the unit doing
    nothing at all rather than taking its default path.
    """
    names = row.get("MatchName")
    if isinstance(names, str):
        names = [names]
    wanted = {n for n in (names or ()) if isinstance(n, str)}
    needed = row.get("NumMatchesNeeded")
    needed = needed if isinstance(needed, int) and needed > 0 else 1

    found = interp._count_units(ctx.team, wanted) if wanted else 0
    target = (
        row.get("ActionToRun") if found >= needed else row.get("ActionToRunIfNoMatch")
    )
    if isinstance(target, str) and target:
        interp.schedule(target, ctx, tick, row.get("ActionDelay", 0))


def _handle_clone(
    interp: ActionInterpreter, row: Mapping[str, Any], ctx: ActionContext, tick: int
) -> None:
    """``ActionClone``: duplicate the friendly troop this ran on.

    The Clone spell acts per unit rather than at a point, which is why it
    arrives here through the area effect's ``OnHitAction`` with the touched
    unit as its source.

    The globals settle the details the action row does not carry:
    ``CLONE_DISTANCE_Y`` (250) offsets the copy a quarter tile so it is not
    perfectly overlapped with its original, ``CLONE_PRESERVE_SHIELD`` keeps a
    Dark Prince's shield -- the reason cloning one is worth doing -- and
    ``CLONE_CLONED_UNITS`` is False, so a clone cannot itself be cloned.

    A clone's single hitpoint is the one part not in the data. It is not in the
    action row, the buff, or any global in this build, so it is applied as the
    card's known behaviour and recorded in reference/anchors.json rather than
    presented as something the files said.
    """
    source = ctx.source
    if source is None or source.dead or source.is_clone:
        return
    spec = source.spec
    if spec is None or source.kind is not EntityKind.TROOP:
        return
    copy = interp._clone(source)
    cloned_action = row.get("OnClonedAction")
    if copy is not None and isinstance(cloned_action, (str, Mapping)) and cloned_action:
        # ``SpawnCloneBufAction`` puts ``BUFF.Clone`` on the copy for 500ms --
        # speed, hit speed and spawn speed all at -100. That is the hologram
        # settling: a clone cannot act for half a second after it appears, and
        # running the node on the *copy* rather than on the original is what
        # makes it the copy that waits.
        interp.run(
            cloned_action,
            ActionContext(team=copy.team, x=copy.x, y=copy.y, source=copy),
            tick,
        )


def _handle_kill(
    interp: ActionInterpreter, row: Mapping[str, Any], ctx: ActionContext, tick: int
) -> None:
    if ctx.source is not None and not ctx.source.dead:
        ctx.source.kill()


def _handle_run_action_at_health(
    interp: ActionInterpreter, row: Mapping[str, Any], ctx: ActionContext, tick: int
) -> None:
    """``ActionRunActionAtHealth``: fire actions as the instigator's hitpoints
    cross listed thresholds.

    ``Actions`` and ``HealthPercentages`` are parallel arrays. GoblinGiant's
    evolution lists 50% three times over because three unrelated actions -- a
    spawner and two cosmetic effects -- all trigger together at that one
    threshold; MovingCannon and GoblinDemolisher each list a single pair.

    There is no "hitpoints changed" event in this engine, so this polls once a
    tick rather than waiting on one. That is bounded rather than expensive: the
    moment every threshold has fired, or the instigator has died, it stops
    rescheduling itself instead of polling for the rest of the match.

    Each threshold fires at most once, tracked in ``ctx.variables`` under a key
    scoped to this node's own name -- necessary because the context is shared
    with whatever else is running on the same instigator, and two different
    ``ActionRunActionAtHealth`` nodes on one unit must not share state.
    """
    source = ctx.source
    if source is None or source.dead:
        return
    stop_if = row.get("ForceStopIfTrue")
    if isinstance(stop_if, str) and stop_if.strip():
        try:
            if evaluate_expression(stop_if, ctx.expression_scope()):
                return
        except ExpressionError:
            interp.unsupported[f"<condition:{stop_if}>"] += 1

    actions = row.get("Actions") or ()
    if isinstance(actions, str):
        actions = [actions]
    percentages = row.get("HealthPercentages") or ()

    node_id = row.get("Name") if isinstance(row.get("Name"), str) else str(id(row))
    fired: set[int] = ctx.variables.setdefault(f"<health-fired:{node_id}>", set())

    max_hp = source.max_hitpoints or 1
    hp_percent = source.hitpoints * 100 // max_hp
    pending = False
    for index, sub_action in enumerate(actions):
        if index in fired or not isinstance(sub_action, (str, Mapping)) or not sub_action:
            continue
        threshold = percentages[index] if index < len(percentages) else None
        if not isinstance(threshold, int):
            continue
        if hp_percent <= threshold:
            fired.add(index)
            interp.run(sub_action, ctx, tick)
        else:
            pending = True

    if pending:
        # Re-resolved by the row itself rather than by name, so an inline
        # node (one with no ``Name`` of its own) keeps polling correctly too.
        interp.schedule(row, ctx, tick, 0)


def _int_list(value: Any) -> list[int]:
    if isinstance(value, int) and not isinstance(value, bool):
        return [value]
    if isinstance(value, (list, tuple)):
        return [v for v in value if isinstance(v, int) and not isinstance(v, bool)]
    return []


def _action_list(value: Any) -> list["str | Mapping[str, Any]"]:
    if isinstance(value, (str, Mapping)):
        return [value] if value else []
    if isinstance(value, (list, tuple)):
        return [v for v in value if isinstance(v, Mapping) or (isinstance(v, str) and v)]
    return []


def _node_key(row: Mapping[str, Any], prefix: str) -> str:
    """A context key scoped to one node.

    Named where the node is named; inline nodes have no name and fall back to
    identity, which is enough because an inline node is reached through exactly
    one parent.
    """
    name = row.get("Name")
    return f"<{prefix}:{name if isinstance(name, str) else id(row)}>"


def _carried_variables(ctx: ActionContext, owner: int) -> dict[str, Any]:
    """What a per-target child context inherits from the effect that spawned it.

    The caster's power ladder, because anything the chain applies scales on the
    *spell's* rarity and level rather than on whatever it happened to land on;
    and who to credit the damage to. Nothing else, so a parent's bookkeeping
    keys (a laser ball's armed flag, a health threshold's fired set) stay with
    the parent.
    """
    carried: dict[str, Any] = {"owner": owner}
    for key in ("buff_rarity", "buff_level"):
        if key in ctx.variables:
            carried[key] = ctx.variables[key]
    return carried


def _handle_laser_ball(
    interp: ActionInterpreter, row: Mapping[str, Any], ctx: ActionContext, tick: int
) -> None:
    """``ActionLaserBall``: Dark Magic, and the whole of what the card does.

    ``AEO.DarkMagicAOE`` declares ``HitsAir: false`` and ``HitsGround: false``,
    so the cloud itself touches nothing; every point of damage the spell deals
    comes through this node, written inline inside the area effect's
    ``OnStartingAction``.

    What it does, read off the row and confirmed against the card's own stat
    sheet in ``STATS.dark_magic``:

    *   Every ``HitFrequency`` (1200ms), starting ``FirstHitDelay`` (1000ms)
        after the node runs, it re-scans ``DetectionRadius`` (2.5 tiles) for
        everything ``HitFilter`` accepts.
    *   **How many it finds decides how hard each one is hit.**
        ``MaxUnitPerActionList`` is ``[1, 4]`` against three entries in
        ``OnDetectedUnitActionList``: one target takes the first tier, two to
        four take the second, five or more take the third. The stat sheet
        settles that this is a count-based tier rather than a per-target rank
        -- the three display strings are ``TID_DAMAGE_WITH_SINGLE_TARGET``,
        ``TID_DAMAGE_WITH_MIN_MAX_TARGETS`` and
        ``TID_DAMAGE_WITH_MIN_OR_MORE_TARGETS``, i.e. "damage with a single
        target" and "damage with 2-4 targets", not "damage to the first
        target". (The tags are named ``damage_first_target`` /
        ``damage_mid_targets``, which reads the other way; the TIDs they are
        displayed through do not.)
    *   The chosen tier's action then runs on **each** unit found, and each
        tier is a one-shot damage buff written inline
        (``DarkMagicAOE_Damage_lv1/2/3``), which is why the spell punishes a
        lone tank far harder than a swarm.

    The scan repeats until the area effect that owns it expires -- 4000ms, so
    three applications at 1500, 2700 and 3900ms.
    """
    source = ctx.source
    if source is None or source.dead:
        return

    armed_key = _node_key(row, "laser")
    if not ctx.variables.get(armed_key):
        ctx.variables[armed_key] = True
        delay = row.get("FirstHitDelay")
        if isinstance(delay, int) and delay > 0:
            interp.schedule(row, ctx, tick, delay)
            return

    detection = row.get("DetectionRadius")
    radius = milli_tiles(detection) if isinstance(detection, int) else 0
    found = interp.objects_in_radius(
        ctx, source.x, source.y, radius, row.get("HitFilter")
    )
    tiers = _action_list(row.get("OnDetectedUnitActionList"))
    if found and tiers:
        bounds = _int_list(row.get("MaxUnitPerActionList"))
        tier = len(tiers) - 1
        for index, bound in enumerate(bounds):
            if len(found) <= bound:
                tier = min(index, len(tiers) - 1)
                break
        action = tiers[tier]
        for victim in found:
            interp.run(
                action,
                ActionContext(
                    team=ctx.team,
                    x=victim.x,
                    y=victim.y,
                    source=victim,
                    variables=_carried_variables(ctx, owner=source.id),
                ),
                tick,
            )

    interval = row.get("HitFrequency")
    if isinstance(interval, int) and interval > 0:
        # One step at a time rather than unrolled, so the scan stops with the
        # cloud that owns it -- the dead-instigator guard in `drain` is what
        # ends it, exactly as it ends a destroyed hut's spawn cycle.
        interp.schedule(row, ctx, tick, interval)


def _handle_air_to_ground(
    interp: ActionInterpreter, row: Mapping[str, Any], ctx: ActionContext, tick: int
) -> None:
    """``ActionAirToGround``: pull the entity this ran on down to the ground.

    Three cards use it and all three mean the same thing -- for
    ``TotalDuration`` this unit is a ground unit. Vines holds what it catches
    down for 2000ms, the Hunter evolution's net for 3000ms, and the Royal Hogs
    evolution lands its jump permanently (999999ms) and then runs
    ``ActionOnGround``.

    The consequence that matters is targeting, not height: a netted Minion can
    be hit by everything that only ``AttacksGround``, and it collides with
    ground troops while it is down.
    """
    source = ctx.source
    if source is None or source.dead:
        return
    interp._set_grounded(source, interp.clock.ticks(row.get("TotalDuration")))
    landed = row.get("ActionOnGround")
    if isinstance(landed, (str, Mapping)) and landed:
        interp.schedule(landed, ctx, tick, row.get("TransitionDuration", 0))


#: How ``TargetSelectionMode`` orders the candidates a shape found. Both modes
#: fall back to entity id, which is this engine's standing deterministic
#: tiebreak -- two units at identical distance or identical health must not
#: depend on iteration order.
_SELECTION_MODES = {
    "Closest": lambda origin: (
        lambda e: (distance(origin[0], origin[1], e.x, e.y), e.id)
    ),
    "HighestCurrentHpIncludeShields": lambda origin: (
        lambda e: (-(e.hitpoints + e.shield), e.id)
    ),
}


def _handle_shape_prio(
    interp: ActionInterpreter, row: Mapping[str, Any], ctx: ActionContext, tick: int
) -> None:
    """``ActionRunActionListOnObjectsInShapeWithPrio``: run a list of actions
    across the best targets in a shape, one action per target.

    Vines is the card built on it: a 2.5-tile circle, the three enemies with
    the most current hitpoints (``HighestCurrentHpIncludeShields``, shields
    counted), and the same snare group fired at each of them 0ms, 50ms and
    150ms apart. ``Actions`` and ``Delays`` are parallel arrays and
    ``OncePerTarget`` is what makes them mean *three different* targets rather
    than three hits on one -- which is exactly the ``total_hit_count`` of 3 the
    card displays.

    ``ActionOnSelfWhenTriggered`` fires back on the instigator when anything
    was found; ``WaitForTarget`` (the champion charges, not Vines) keeps
    looking until something enters the shape.
    """
    radius = interp.shape_radius(row.get("Shape"))
    if radius is None:
        return
    source = ctx.source
    origin = (source.x, source.y) if source is not None else (ctx.x, ctx.y)
    found = interp.objects_in_radius(
        ctx, origin[0], origin[1], radius, row.get("TargetFilter")
    )
    mode = str(row.get("TargetSelectionMode", "Closest"))
    key = _SELECTION_MODES.get(mode)
    if key is None:
        interp.unsupported[f"<selection:{mode}>"] += 1
        key = _SELECTION_MODES["Closest"]
    found.sort(key=key(origin))

    if not found:
        if row.get("WaitForTarget") is True and source is not None and not source.dead:
            # Bounded by the instigator's life, the same way an unbounded
            # ActionInterval is: a champion who never finds a target simply
            # stops looking when it dies.
            interp.schedule(row, ctx, tick, 0)
        return

    actions = _action_list(row.get("Actions"))
    delays = _int_list(row.get("Delays"))
    for index, action in enumerate(actions):
        if index >= len(found):
            # Fewer targets than actions: the surplus actions have nothing to
            # run on. Vines against a lone Giant snares it once, not three
            # times -- OncePerTarget.
            break
        victim = found[index]
        interp.schedule(
            action,
            ActionContext(
                team=ctx.team,
                x=victim.x,
                y=victim.y,
                source=victim,
                variables=dict(ctx.variables),
            ),
            tick,
            delays[index] if index < len(delays) else 0,
        )
    on_self = row.get("ActionOnSelfWhenTriggered")
    if isinstance(on_self, (str, Mapping)) and on_self:
        interp.schedule(on_self, ctx, tick, 0)
    on_finished = row.get("OnFinishedAction")
    if isinstance(on_finished, (str, Mapping)) and on_finished:
        interp.schedule(on_finished, ctx, tick, max(delays) if delays else 0)


def _handle_change_data(
    interp: ActionInterpreter, row: Mapping[str, Any], ctx: ActionContext, tick: int
) -> None:
    """``ActionChangeGameObjectData``: swap what a live unit *is*.

    Two standard cards turn on it, both at 50% health. Cannon Cart's
    ``MovingCannon`` becomes ``BrokenCannon`` -- ``IsBuilding = true``, no
    ``Speed``, ``LifeTime = 30000`` -- so it stops moving, gains a 30-second
    clock and starts being a legal target for every building-targeting troop
    on the board. Goblin Demolisher becomes ``GoblinDemolisher_kamikaze_form``
    -- twice the speed, melee range, ``Kamikaze``, ``TargetOnlyBuildings`` --
    which is the whole reason to play the card.

    Current hitpoints are carried across rather than reset. The Tombstone hero
    is the evidence: its swap is the only one that chains a
    ``Tombstone_hero_ResetHealthValue`` afterwards, which would be redundant if
    the swap reset health by itself. Both cards above transform at half health
    and must stay at half health.

    ``NewProjectileData`` -- the other spelling, used by the Executioner and
    Snowball evolutions -- is **not** implemented and is recorded as a gap: a
    unit's damage in this engine is resolved from its projectile at spec-build
    time, so swapping the projectile means rebuilding the damage as well, and
    guessing at that would be worse than leaving it visible.
    """
    source = ctx.source
    if source is None or source.dead:
        return
    new_character = row.get("NewCharacterData")
    if not isinstance(new_character, str) or not new_character:
        if row.get("NewProjectileData"):
            interp.unsupported["<changedata:projectile>"] += 1
        return
    if interp._change_data(source, new_character, row.get("ResetTarget") is True):
        _handle_next(interp, row, ctx, tick)


def _handle_deal_damage(
    interp: ActionInterpreter, row: Mapping[str, Any], ctx: ActionContext, tick: int
) -> None:
    """``ActionDealDamage``: hit the entity this ran on for a flat amount.

    ``BaseDamageAmount`` where the row carries one (the Three Musketeers
    rework's bayonet, the Firecracker projectile's deflect), or the amount the
    chain put in context -- which is how Ronin's parry pays back what it
    blocked, since ``ronin_reflect_damage`` carries no amount of its own.
    """
    target = ctx.source
    if target is None or target.dead:
        return
    amount = row.get("BaseDamageAmount")
    if not isinstance(amount, int) or amount <= 0:
        amount = ctx.variables.get("damage_amount")
    if not isinstance(amount, int) or amount <= 0:
        interp.unsupported["<dealdamage:no-amount>"] += 1
        return
    interp._deal_damage(int(ctx.variables.get("damage_source", 0)), target, amount)


def _handle_target_indicator_attack(
    interp: ActionInterpreter, row: Mapping[str, Any], ctx: ActionContext, tick: int
) -> None:
    """``ActionTargetIndicatorAttack``: Goblin Machine's rocket.

    A second weapon on its own clock, independent of the melee swing the stat
    columns describe. Every ``AttackCooldown`` (4000ms) it looks for a target
    between ``MinimumRange`` (2.5 tiles) and ``Range`` (5 tiles) that
    ``TargetFilter`` accepts, marks the ground under it with ``TargetAoE``, and
    ``LoadTime`` (1500ms) later fires ``Projectile`` at **that spot** rather
    than at the unit.

    The ground-aiming is the point of the card and the reason the indicator
    exists at all: a shot that followed its target would need no marker, and
    ``GoblinMachineRocketProjectile`` carries no ``Homing`` flag. Walking out
    of the circle is how the rocket is dodged.

    Two fields are deliberately not modelled, on the same grounds the ordinary
    attack path already ignores their equivalents: ``AttackDelay`` (1000ms) and
    ``ProjectileOffsetToCharacterLookDirection`` (-1200) place the projectile
    within the firing *animation*, and this engine spawns every other
    projectile on the tick its swing resolves, from the attacker's centre.
    """
    source = ctx.source
    if source is None or source.dead:
        return

    key = _node_key(row, "indicator")
    stage = ctx.variables.get(key)
    cooldown = interp.clock.ticks(row.get("AttackCooldown"))

    if stage == "fire":
        ctx.variables[key] = None
        aim = ctx.variables.get(f"{key}:aim")
        projectile = row.get("Projectile")
        if aim is not None and isinstance(projectile, str) and projectile:
            interp._fire_projectile(source, projectile, aim[0], aim[1])
            shot_action = row.get("OnProjectileShootAction")
            if isinstance(shot_action, (str, Mapping)) and shot_action:
                interp.schedule(shot_action, ctx, tick, 0)
        # The cooldown is measured lock-to-lock, so the gap between rockets
        # comes out at exactly ``AttackCooldown`` -- which is the four seconds
        # the card displays as ``cooldown``. Counting it from the shot instead
        # would put 5.5 seconds between rockets and make the printed stat a
        # lie.
        interp.schedule(
            row, ctx, tick,
            max(0, int(row.get("AttackCooldown", 0) or 0) - int(row.get("LoadTime", 0) or 0)),
        )
        return

    maximum = row.get("Range")
    minimum = row.get("MinimumRange")
    maximum = milli_tiles(maximum) if isinstance(maximum, int) else 0
    minimum = milli_tiles(minimum) if isinstance(minimum, int) else 0
    candidates = [
        entity
        for entity in interp.objects_in_radius(
            ctx, source.x, source.y, maximum, row.get("TargetFilter")
        )
        if distance(source.x, source.y, entity.x, entity.y) >= minimum
    ]
    if not candidates:
        # Nothing in the band. Look again next tick rather than burning the
        # cooldown: the machine is not on a timer it wastes, it is waiting.
        interp.schedule(row, ctx, tick, 0)
        return

    candidates.sort(key=lambda e: (distance(source.x, source.y, e.x, e.y), e.id))
    aim = candidates[0]
    ctx.variables[key] = "fire"
    ctx.variables[f"{key}:aim"] = (aim.x, aim.y)
    marker = row.get("TargetAoE")
    if isinstance(marker, str) and marker:
        interp._place_area(ctx.team, marker, aim.x, aim.y, source)
    indication = row.get("TargetStartIndicationAction")
    if isinstance(indication, (str, Mapping)) and indication:
        interp.schedule(indication, ctx, tick, row.get("TargetIndicatorDelay", 0))
    interp.schedule(row, ctx, tick, row.get("LoadTime", 0))
    if cooldown <= 0:
        interp.unsupported["<indicator:no-cooldown>"] += 1


def _handle_counter(
    interp: ActionInterpreter, row: Mapping[str, Any], ctx: ActionContext, tick: int
) -> None:
    """``ActionCounter``: Ronin's parry.

    One node, one card. ``ronin_parry`` is ``DeployActive``, so it is armed
    from the moment he lands, and once armed the next attack that reaches him
    is turned around: ``DefenseScalar`` (0) is the percentage of the incoming
    damage he actually takes, ``DamageScalar`` (200) is the percentage handed
    back to whoever swung -- and the card's own stat sheet displays that 200 as
    ``parry_damage`` in ``PERCENT_NOSIGN``, which is what fixes it as a
    percentage of the blocked hit rather than a flat number.

    Parrying then runs two chains: ``SelfAction`` on Ronin (the 3500ms
    cooldown tag) and ``InstigatorAction`` on the attacker, which is a 500ms
    stun at +50ms followed by the reflected damage at +300ms.

    Registration is all that happens here; the parry itself has to fire when a
    hit lands, which only the battle sees.
    """
    source = ctx.source
    if source is None or source.dead:
        return
    interp._arm_counter(source, row, ctx)


_HANDLERS: dict[str, Handler] = {
    "ActionGroup": _handle_group,
    "ActionSpawn": _handle_spawn,
    "ActionSpawnToLocation": _handle_spawn,
    "ActionSetVariable": _handle_set_variable,
    "ActionWithDuration": _handle_next,
    "ActionSelect": _handle_select,
    "ActionFilter": _handle_next,
    "ActionWaitToActivate": _handle_next,
    "ActionActivateOnCardDeploy": _handle_next,
    "ActionInterval": _handle_interval,
    "ActionKill": _handle_kill,
    "ActionClone": _handle_clone,
    "ActionRunIfGameObjectExists": _handle_run_if_exists,
    "ActionGoblinHutLifeState": _handle_hut_life_state,
    "ActionRunActionAtHealth": _handle_run_action_at_health,
    "ActionLaserBall": _handle_laser_ball,
    "ActionAirToGround": _handle_air_to_ground,
    "ActionRunActionListOnObjectsInShapeWithPrio": _handle_shape_prio,
    "ActionChangeGameObjectData": _handle_change_data,
    "ActionDealDamage": _handle_deal_damage,
    "ActionCounter": _handle_counter,
    "ActionTargetIndicatorAttack": _handle_target_indicator_attack,
}
