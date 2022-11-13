from asyncio import Future, get_event_loop
from dataclasses import dataclass
from typing import Awaitable, Callable, Generic, Optional, TypeVar


Event = TypeVar("Event")


@dataclass(frozen=True, slots=True)
class Listenable(Generic[Event]):
    listen: Callable[[Callable[[Event], None]], Callable[[], None]]


@dataclass(frozen=True, slots=True)
class Emittable(Listenable, Generic[Event]):
    emit: Callable[[Event], None]


def emittable(
    start: Optional[Callable[[Callable[[Event], None]], Callable[[], None]]] = None,
) -> Emittable[Event]:
    listeners: set[Callable[[Event], None]] = set()

    stop: Callable[[], None]

    def listen(handler: Callable[[Event], None]) -> Callable[[], None]:
        nonlocal stop
        if start is not None and len(listeners) == 0:
            stop = start(emit)

        listeners.add(handler)

        def unlisten():
            listeners.remove(handler)
            if start is not None and len(listeners) == 0:
                stop()

        return unlisten

    def emit(event: Event):
        for handler in listeners.copy():
            handler(event)

    return Emittable(listen=listen, emit=emit)


def listenable(
    start: Optional[Callable[[Callable[[Event], None]], Callable[[], None]]] = None,
) -> Listenable[Event]:
    emitter = emittable(start)
    return Listenable(listen=emitter.listen)


def successor(emitter: Listenable[Event]) -> Awaitable[Event]:
    future = Future()

    def handler(event: Event):
        unlisten()
        future.set_result(event)

    unlisten = emitter.listen(handler)
    return future


MappedEvent = TypeVar("MappedEvent")


def mapped(
    emitter: Listenable[Event], fn: Callable[[Event], MappedEvent]
) -> Listenable[MappedEvent]:
    def start(emit):
        def handler(event: Event):
            emit(fn(event))

        unlisten = emitter.listen(handler)
        return unlisten

    return listenable(start)
