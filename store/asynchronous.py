from asyncio import BaseEventLoop, Task, TimerHandle
from datetime import timedelta
from functools import partial
from operator import lshift
from typing import Any, Awaitable, Callable, Optional

from ..fallibility import NONE, MatchesNone, MatchesSome, Option

from . import Args, DerivedValue, Readable, Value, readable


def async_subscribe(
    store: Readable[Value],
    async_callback: Callable[[Value], Awaitable[None]],
    *,
    loop: BaseEventLoop,
):
    def callback(value: Value):
        loop.create_task(async_callback(value))

    return store.subscribe(callback)


def async_subscribe_exclusive(
    store: Readable[Value],
    async_callback: Callable[[Value], Awaitable[None]],
    *,
    loop: BaseEventLoop,
):
    task: Option[Task] = NONE()

    def callback(value: Value):
        match task.take().to_matchable():
            case MatchesSome(task_):
                task_.cancel()

        task.insert(loop.create_task(async_callback(value)))

    # TODO: it's ambiguous if the current task should be cancelled when unsubscribing
    return store.subscribe(callback)



# TODO: extract elsewhere
def no_op():
    pass



def derived_with_time(
    stores: list[Readable[Any]],
    derivation: Callable[Args, DerivedValue],
    get_max_period: Callable[[], timedelta],
    *,
    loop: BaseEventLoop,
) -> Readable[DerivedValue]:
    n = len(stores)

    def start(set_: Callable[[DerivedValue], None]):
        values: list[Optional[Any]] = [None] * n
        pending = sum(map(partial(lshift, 1), range(n)))

        timer_handle: Option[TimerHandle] = NONE()

        def calculate_then_set_and_delay():
            result = derivation(*values)
            set_(result)

            match timer_handle.take().to_matchable():
                case MatchesSome(handle):
                    handle.cancel()

            delay = get_max_period()
            timer_handle.insert(
                loop.call_later(
                    delay.total_seconds(), calculate_then_set_and_delay
                )
            )

        if n == 0:
            calculate_then_set_and_delay()

            return no_op

        def subscribe_to_store(i_store: tuple[int, Readable[Any]]):
            (i, store) = i_store

            def callback(value: Any):
                values[i] = value
                nonlocal pending
                if pending:
                    pending &= ~(1 << i)
                if not pending:
                    calculate_then_set_and_delay()

            unsubscribe = store.subscribe(callback)
            return unsubscribe

        unsubscribers: list[Callable[[], None]] = list(
            map(subscribe_to_store, enumerate(stores))
        )

        def stop():
            match timer_handle.take().to_matchable():
                case MatchesSome(handle):
                    handle.cancel()

            for unsubscribe in unsubscribers:
                unsubscribe()

        return stop

    return readable(None, start)  # type: ignore


# TODO: a version that raises an exception when the state becomes undesirable
def satisfies(
    store: Readable[Value],
    predicate: Callable[[Value], bool],
    *,
    loop: BaseEventLoop
) -> Awaitable[Value]:
    future = loop.create_future()

    satisfied = False
    unsubscribe: Option[Callable[[], None]] = NONE()

    def callback(value: Value):
        nonlocal satisfied

        if satisfied:
            return

        if predicate(value):
            satisfied = True

            match unsubscribe.to_matchable():
                case MatchesSome(unsubscriber):
                    unsubscriber()

                case MatchesNone():
                    # Wait a tick for it to be possible to unsubscribe
                    loop.call_soon(lambda: unsubscribe.unwrap()())

            future.set_result(value)

    unsubscribe.insert(store.subscribe(callback))
    return future
