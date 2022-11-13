from asyncio import Queue
from collections import deque
from dataclasses import dataclass
from typing import (
    AsyncGenerator,
    Callable,
    Deque,
    Generic,
    Optional,
    TypeVar,
)

from emitter import Listenable, listenable

from . import Value, Readable, readable


def values(store: Readable[Value]) -> AsyncGenerator[Value, None]:
    queue = Queue()

    unsubscribe = store.subscribe(queue.put_nowait)

    async def generator():
        try:
            while True:
                yield await queue.get()
        finally:
            unsubscribe()

    return generator()


@dataclass
class ValueChange(Generic[Value]):
    from_value: Value
    to_value: Value


def changes(store: Readable[Value]) -> Listenable[ValueChange[Value]]:
    def start(emit):
        initialized = False

        last_value: Value

        def run(value):
            nonlocal initialized
            nonlocal last_value

            if initialized:
                emit(ValueChange(from_value=last_value, to_value=value))

            initialized = True
            last_value = value

        unsubscribe = store.subscribe(run)
        return unsubscribe

    return listenable(start)

Total = TypeVar("Total")


def scan(
    store: Readable[Value], reducer: Callable[[Total, Value], Total], seed: Total
) -> Readable[Total]:
    def start(set_: Callable[[Value], None]):
        total = seed

        def callback(value: Value):
            nonlocal total

            total = reducer(total, value)
            set_(total)

        unsubscribe = store.subscribe(callback)
        return unsubscribe

    return readable(seed, start)


def history(
    store: Readable[Value], limit: Optional[int] = None
) -> Readable[deque[Value]]:
    def reducer(total: Deque[Value], value: Value):
        new = total.copy()
        new.append(value)
        return new

    return scan(store, reducer, deque(maxlen=limit))
