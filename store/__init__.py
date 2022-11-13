from dataclasses import dataclass
from functools import partial
from operator import lshift
from typing import Any, Callable, Generic, Optional, ParamSpec, Tuple, TypeVar


Value = TypeVar("Value")
# TODO: remove SetReturn... or should I not?
SetReturn = TypeVar("SetReturn")


@dataclass(frozen=True, slots=True)
class Readable(Generic[Value]):
    subscribe: Callable[[Callable[[Value], None]], Callable[[], None]]


@dataclass(frozen=True, slots=True)
class Writable(Readable[Value], Generic[Value, SetReturn]):
    set: Callable[[Value], SetReturn]


@dataclass(frozen=True, slots=True)
class Updatable(Writable[Value, SetReturn], Generic[Value, SetReturn]):
    update: Callable[[Callable[[Value], Value]], None]


def writable(
    initial_value: Value,
    start: Optional[Callable[[Callable[[Value], None]], Callable[[], None]]] = None,
) -> Updatable[Value, None]:
    value = initial_value

    subscribers: set[Callable[[Value], None]] = set()
    stop: Callable[[], None]

    def subscribe(callback: Callable[[Value], None]) -> Callable[[], None]:
        if start is not None and len(subscribers) == 0:
            nonlocal stop
            stop = start(set_)

        subscribers.add(callback)
        callback(value)

        def unsubscribe():
            subscribers.remove(callback)
            if start is not None and len(subscribers) == 0:
                stop()

        return unsubscribe

    def set_(new_value: Value):
        nonlocal value
        if new_value != value:
            value = new_value

            for callback in subscribers.copy():
                callback(value)

    def update(fn: Callable[[Value], Value]):
        set_(fn(value))

    return Updatable(subscribe=subscribe, set=set_, update=update)



def writable_unchecked(
    initial_value: Value,
    start: Optional[Callable[[Callable[[Value], None]], Callable[[], None]]] = None,
) -> Updatable[Value, None]:
    value = initial_value

    subscribers: set[Callable[[Value], None]] = set()
    stop: Callable[[], None]

    def subscribe(callback: Callable[[Value], None]) -> Callable[[], None]:
        if start is not None and len(subscribers) == 0:
            nonlocal stop
            stop = start(set_)

        subscribers.add(callback)
        callback(value)

        def unsubscribe():
            subscribers.remove(callback)
            if start is not None and len(subscribers) == 0:
                stop()

        return unsubscribe

    def set_(new_value: Value):
        nonlocal value
        value = new_value

        for callback in subscribers.copy():
            callback(value)

    def update(fn: Callable[[Value], Value]):
        set_(fn(value))

    return Updatable(subscribe=subscribe, set=set_, update=update)


def readable(
    initial_value: Value,
    start: Optional[Callable[[Callable[[Value], None]], Callable[[], None]]] = None,
) -> Readable[Value]:
    store = writable(initial_value, start)
    return Readable(subscribe=store.subscribe)


def get(store: Readable[Value]) -> Value:
    value: Value

    def retrieve_value(existing_value: Value):
        nonlocal value
        value = existing_value

    unsubscribe = store.subscribe(retrieve_value)
    unsubscribe()

    return value


DerivedValue = TypeVar("DerivedValue")
Args = ParamSpec("Args")


# TODO: extract elsewhere
def no_op():
    pass


def derived(
    stores: list[Readable[Any]],
    fn: Callable[Args, DerivedValue],
) -> Readable[DerivedValue]:
    n = len(stores)

    def start(set_: Callable[[DerivedValue], None]):
        if n == 0:
            set_(fn())
            return no_op

        values: list[Optional[Any]] = [None] * n
        pending = sum(map(partial(lshift, 1), range(n)))

        def subscribe_to_store(i_store: Tuple[int, Readable[Any]]):
            (i, store) = i_store

            def callback(value: Any):
                values[i] = value
                nonlocal pending
                if pending:
                    pending &= ~(1 << i)
                if not pending:
                    result = fn(*values)
                    set_(result)

            unsubscribe = store.subscribe(callback)
            return unsubscribe

        unsubscribers: list[Callable[[], None]] = list(
            map(subscribe_to_store, enumerate(stores))
        )

        def stop():
            for unsubscribe in unsubscribers:
                unsubscribe()

        return stop

    return readable(None, start)  # type: ignore


def derived_unchecked(
    stores: list[Readable[Any]],
    fn: Callable[Args, DerivedValue],
) -> Readable[DerivedValue]:
    n = len(stores)

    def start(set_: Callable[[DerivedValue], None]):
        if n == 0:
            set_(fn())
            return no_op

        values: list[Optional[Any]] = [None] * n
        pending = sum(map(partial(lshift, 1), range(n)))

        def subscribe_to_store(i_store: Tuple[int, Readable[Any]]):
            (i, store) = i_store

            def callback(value: Any):
                values[i] = value
                nonlocal pending
                if pending:
                    pending &= ~(1 << i)
                if not pending:
                    result = fn(*values)
                    set_(result)

            unsubscribe = store.subscribe(callback)
            return unsubscribe

        unsubscribers: list[Callable[[], None]] = list(
            map(subscribe_to_store, enumerate(stores))
        )

        def stop():
            for unsubscribe in unsubscribers:
                unsubscribe()

        return stop

    return writable_unchecked(None, start)  # type: ignore
