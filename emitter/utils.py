from asyncio import Queue
from collections import deque
from typing import (
    AsyncGenerator,
    Callable,
    Deque,
    Iterable,
    Optional,
    Tuple,
    TypeVar,
)

from enumeration import Data, Enumerant, Variant, get_data, get_variant
from store import Readable, Value, readable
from store.utils import ValueChange

from . import Event, Listenable, emittable, listenable, mapped


def events(emitter: Listenable[Event]) -> AsyncGenerator[Event, None]:
    queue = Queue()

    unlisten = emitter.listen(queue.put_nowait)

    async def generator():
        try:
            while True:
                yield await queue.get()
        finally:
            unlisten()

    return generator()


def latest(emitter: Listenable[Event], initial_value: Event = None) -> Readable[Event]:
    return readable(initial_value, emitter.listen)


def filtered(
    emitter: Listenable[Event], predicate: Callable[[Event], bool]
) -> Listenable[Event]:
    def start(emit: Callable[[Event], None]):
        def handler(event: Event):
            if predicate(event):
                emit(event)

        unlisten = emitter.listen(handler)
        return unlisten

    return listenable(start)


Total = TypeVar("Total")


def scan(
    emitter: Listenable[Event], reducer: Callable[[Total, Event], Total], seed: Total
) -> Readable[Total]:
    def start(set_: Callable[[Event], None]):
        total = seed

        def handler(event: Event):
            nonlocal total

            total = reducer(total, event)
            set_(total)

        unlisten = emitter.listen(handler)
        return unlisten

    return readable(seed, start)


def recent(
    emitter: Listenable[Event], limit: Optional[int] = None
) -> Readable[deque[Event]]:
    def reducer(total: Deque[Event], event: Event):
        new = total.copy()
        new.append(event)
        return new

    return scan(emitter, reducer, deque(maxlen=limit))


Category = TypeVar("Category")


def categorize(
    emitter: Listenable[Event],
    key: Callable[[Event], Category],
    categories: Iterable[Category],
) -> dict[Category, Listenable[Event]]:
    emits: dict[Category, Callable[[Event], None]] = {}
    listenables: dict[Category, Listenable[Event]] = {}

    active_listeners = 0
    unlisten: Callable[[], None]

    def switch(event: Event):
        destination = key(event)
        if destination not in emits:
            raise KeyError(
                f"{destination!r} is not in {categories!r}, so an event for it cannot be emitted"
            )
        emit = emits[destination]
        emit(event)

    def start_subemitter(i: int):
        def start(_emit: Callable[[Event], None]):
            nonlocal active_listeners

            if not active_listeners:
                nonlocal unlisten
                unlisten = emitter.listen(switch)

            active_listeners |= 1 << i

            def stop():
                nonlocal active_listeners
                active_listeners &= ~(1 << i)

                if not active_listeners:
                    unlisten()

            return stop

        return start

    for i, category in enumerate(categories):
        subemitter = emittable(start_subemitter(i))

        emits[category] = subemitter.emit
        listenables[category] = Listenable(listen=subemitter.listen)

    return listenables


def partitioned(
    emitter: Listenable[Event], predicate: Callable[[Event], bool]
) -> Tuple[Listenable[Event], Listenable[Event]]:
    categorized = categorize(emitter, predicate, [False, True])
    return (categorized[False], categorized[True])

# def categorize_enumeration(
#     emitter: Listenable[Enumerant[Variant, Data]], variants: list[Variant]
# ) -> dict[
#     Variant, Listenable[Data]
# ]:  # TODO / NOTE: relies on the consumer to provide types
#     categorized = categorize(emitter, get_variant, variants)
#     return map_values(categorized, lambda emitter: mapped(emitter, get_data))


# TODO: should there be an accompanying joiner / multiplexer / opposite of groupby / opposite of categorize? what's a use case for that?


def get_from_value(value_change: ValueChange[Value]) -> Value:
    return value_change.from_value


def get_to_value(value_change: ValueChange[Value]) -> Value:
    return value_change.to_value


def from_value(changes: Listenable[ValueChange[Value]]) -> Listenable[Value]:
    return mapped(changes, get_from_value)


def to_value(changes: Listenable[ValueChange[Value]]) -> Listenable[Value]:
    return mapped(changes, get_to_value)
