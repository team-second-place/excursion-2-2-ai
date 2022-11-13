
from asyncio import BaseEventLoop, Task
from typing import AsyncIterator, Awaitable, Callable


from ..fallibility import NONE, MatchesSome, Option

from . import Event, Listenable, listenable


def async_listen(
    emitter: Listenable[Event],
    async_handler: Callable[[Event], Awaitable[None]],
    *,
    loop: BaseEventLoop,
):
    def handler(event: Event):
        loop.create_task(async_handler(event))

    return emitter.listen(handler)


def async_listen_exclusive(
    emitter: Listenable[Event],
    async_handler: Callable[[Event], Awaitable[None]],
    *,
    loop: BaseEventLoop,
):
    task: Option[Task] = NONE()

    def handler(event: Event):
        match task.take().to_matchable():
            case MatchesSome(task_):
                task_.cancel()

        task.insert(loop.create_task(async_handler(event)))

    # TODO: it's ambiguous if the current task should be cancelled when unlistening
    return emitter.listen(handler)


def async_iterator_to_emitter(events: AsyncIterator[Event], *, loop: BaseEventLoop) -> Listenable[Event]:
    def start(emit: Callable[[Event], None]):
        async def iterate():
            async for event in events:
                emit(event)

        task = loop.create_task(iterate())
        return task.cancel

    return listenable(start)
