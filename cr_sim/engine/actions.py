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

Coverage is deliberately partial and deliberately loud. The structural node
types are implemented; the long tail of one-off classes
(``ActionMegaKnightUppercut``, ``ActionSoulDrain``) is not, and every one that
is encountered is recorded in :attr:`ActionInterpreter.unsupported` so a card
that quietly does nothing shows up as a name rather than as a mystery.
"""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from ..data.source import LogicData, UnknownEntity
from .constants import TickClock
from .entity import Entity, EntityKind, Team

__all__ = [
    "ActionContext",
    "ActionInterpreter",
    "ExpressionError",
    "evaluate_expression",
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
        scope: dict[str, Any] = {
            "x": self.x // 18,
            "y": self.y // 18,
            "map_width": MAP_WIDTH_MILLI_TILES,
            "team_index": int(self.team),
            "is_clone": False,
            "is_deploying": bool(self.source is not None and self.source.is_deploying),
            "is_moving": False,
            "hp": self.source.hitpoints if self.source is not None else 0,
        }
        scope.update(self.variables)
        return scope


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
    action: str
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
        "_pending", "unsupported", "_cache",
    )

    def __init__(
        self,
        data: LogicData,
        clock: TickClock,
        spawn: Callable[..., list[Entity]],
        clone: Callable[[Entity], Entity | None] | None = None,
        place_area: Callable[..., Any] | None = None,
        count_units: Callable[[Team, set[str]], int] | None = None,
    ) -> None:
        self.data = data
        self.clock = clock
        #: Injected rather than reached for, so this module never imports the
        #: battle and the two can be tested apart.
        self._spawn = spawn
        self._clone = clone or (lambda entity: None)
        self._place_area = place_area or (lambda *a, **k: None)
        self._count_units = count_units or (lambda team, names: 0)
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

    def schedule(self, name: str, context: ActionContext, tick: int, delay_ms: Any = 0) -> None:
        self._pending.append(
            _Pending(tick + self.clock.ticks(delay_ms), name, context)
        )

    def start(self, name: str, context: ActionContext, tick: int) -> None:
        """Begin an action graph, honouring the entry node's own start delay.

        ``ActionDelay`` on the entry node is a real part of the behaviour: the
        reworked Goblin Hut waits 1000ms before its first Spear Goblin.
        Running the graph immediately makes every such card produce its first
        output on the tick it lands.
        """
        row = self.resolve(name)
        delay = row.get("ActionDelay", 0) if row is not None else 0
        if isinstance(delay, int) and delay > 0:
            self.schedule(name, context, tick, delay)
        else:
            self.run(name, context, tick)

    def run(self, name: str, context: ActionContext, tick: int) -> None:
        """Execute one action now, scheduling whatever it defers."""
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
        if not self._passes(row, context):
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

    def _passes(self, row: Mapping[str, Any], context: ActionContext) -> bool:
        """Gate an action on its own condition, where it carries one."""
        for key in ("ExecuteIfTrue", "Condition"):
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
        # Fixed relative offsets, for the nodes that use them instead.
        rel_x, rel_y = row.get("RelativeX"), row.get("RelativeY")
        if isinstance(rel_x, int):
            x += rel_x * 18
        if isinstance(rel_y, int):
            y += rel_y * 18 * _team_y_direction(int(context.team))
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
        if not isinstance(sub, str):
            continue
        delay = delays[index] if index < len(delays) else 0
        interp.schedule(sub, ctx, tick, delay)


def _handle_spawn(
    interp: ActionInterpreter, row: Mapping[str, Any], ctx: ActionContext, tick: int
) -> None:
    """``ActionSpawn`` / ``ActionSpawnToLocation``: put a character on the board."""
    what = row.get("SpawnData")
    if not isinstance(what, str) or not what:
        return
    spawn_type = str(row.get("SpawnType", "CharacterType"))
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
    """Nodes whose only job is to hand off to another after a delay."""
    for key in ("NextAction", "ActionToExecute"):
        target = row.get(key)
        if isinstance(target, str) and target:
            interp.schedule(target, ctx, tick, row.get("ActionDelay", 0))


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
    interp._clone(source)


def _handle_kill(
    interp: ActionInterpreter, row: Mapping[str, Any], ctx: ActionContext, tick: int
) -> None:
    if ctx.source is not None and not ctx.source.dead:
        ctx.source.kill()


_HANDLERS: dict[str, Handler] = {
    "ActionGroup": _handle_group,
    "ActionSpawn": _handle_spawn,
    "ActionSpawnToLocation": _handle_spawn,
    "ActionSetVariable": _handle_set_variable,
    "ActionWithDuration": _handle_next,
    "ActionSelect": _handle_next,
    "ActionFilter": _handle_next,
    "ActionWaitToActivate": _handle_next,
    "ActionActivateOnCardDeploy": _handle_next,
    "ActionInterval": _handle_interval,
    "ActionKill": _handle_kill,
    "ActionClone": _handle_clone,
    "ActionRunIfGameObjectExists": _handle_run_if_exists,
    "ActionGoblinHutLifeState": _handle_hut_life_state,
}
