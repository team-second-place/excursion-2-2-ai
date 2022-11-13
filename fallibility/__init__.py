# MIT License

# Copyright (c) 2018-2022 Peijun Ma

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
This module contains the Option and Result classes.
"""

from dataclasses import dataclass
from typing import Any, Callable, Generic, Optional, Protocol, TypeVar, Union


S = TypeVar("S")

O = TypeVar("O")
E = TypeVar("E")

U = TypeVar("U")


@dataclass
class MatchesNone:
    pass


@dataclass
class MatchesSome(Generic[S]):
    value: S


class SupportsDunderLT(Protocol):
    def __lt__(self, __other: object) -> bool:
        ...


class SupportsDunderGT(Protocol):
    def __gt__(self, __other: object) -> bool:
        ...


class SupportsDunderLE(Protocol):
    def __le__(self, __other: object) -> bool:
        ...


class SupportsDunderGE(Protocol):
    def __ge__(self, __other: object) -> bool:
        ...


class Option(Generic[S]):
    """
    :py:class:`Option` represents an optional value. Every :py:class:`Option`
    is either ``Some`` and contains a value, or :py:data:`NONE` and
    does not.

    To create a ``Some`` value, please use :py:meth:`Option.Some` or :py:func:`Some`.

    To create a :py:data:`NONE` value, please use :py:meth:`Option.NONE` :py:func:`NONE`.

    To let :py:class:`Option` guess the type of :py:class:`Option` to create,
    please use :py:func:`maybe`.

    Calling the ``__init__``  method directly will raise a ``TypeError``.

    Examples:
        >>> Option.Some(1)
        Some(1)
        >>> Option.NONE()
        NONE
        >>> maybe(1)
        Some(1)
        >>> maybe(None)
        NONE
    """

    __slots__ = ("_val", "_is_some")

    def __init__(self, value: S, is_some: bool, *, _force: bool = False) -> None:
        if not _force:
            raise TypeError(
                "Cannot directly initialize, "
                "please use one of the factory functions instead."
            )
        self._val = value
        self._is_some = is_some

    @classmethod
    def Some(cls, val: S) -> "Option[S]":
        """Some value ``val``."""
        return cls(val, True, _force=True)

    @classmethod
    def NONE(cls) -> "Option[S]":
        """No Value."""
        return NONE()

    # TODO: should I remove this or does rust also work like this?
    def __bool__(self) -> bool:
        """
        Returns the truth value of the :py:class:`Option` based on its value.

        Returns:
            True if the :py:class:`Option` is ``Some`` value, otherwise False.

        Examples:
            >>> bool(Some(False))
            True
            >>> bool(NONE())
            False
        """
        return self._is_some

    def is_some(self) -> bool:
        """
        Returns ``True`` if the option is a ``Some`` value.

        Examples:
            >>> Some(0).is_some
            True
            >>> NONE().is_some
            False
        """
        return self._is_some

    def is_none(self) -> bool:
        """
        Returns ``True`` if the option is a :py:data:`NONE` value.

        Examples:
            >>> Some(0).is_none()
            False
            >>> NONE().is_none()
            True
        """
        return not self._is_some

    def expect(self, msg: object) -> S:
        """
        Unwraps the option. Raises an exception if the value is :py:data:`NONE`.

        Args:
            msg: The exception message.

        Returns:
            The wrapped value.

        Raises:
            ``ValueError`` with message provided by ``msg`` if the value is :py:data:`NONE`.

        Examples:
            >>> Some(0).expect('sd')
            0
            >>> try:
            ...     NONE().expect('Oh No!')
            ... except ValueError as e:
            ...     print(e)
            Oh No!
        """
        if self._is_some:
            return self._val
        raise ValueError(msg)

    def unwrap(self) -> S:
        """
        Returns the value in the :py:class:`Option` if it is ``Some``.

        Returns:
            The ```Some`` value of the :py:class:`Option`.

        Raises:
            ``ValueError`` if the value is :py:data:`NONE`.

        Examples:
            >>> Some(0).unwrap()
            0
            >>> try:
            ...     NONE().unwrap()
            ... except ValueError as e:
            ...     print(e)
            Value is NONE.
        """
        if self._is_some:
            return self._val
        raise ValueError("Value is NONE.")

    def unwrap_or(self, default: S) -> S:
        """
        Returns the contained value or ``default``.

        Args:
            default: The default value.

        Returns:
            The contained value if the :py:class:`Option` is ``Some``,
            otherwise ``default``.

        Notes:
            If you wish to use a result of a function call as the default,
            it is recommnded to use :py:meth:`unwrap_or_else` instead.

        Examples:
            >>> Some(0).unwrap_or(3)
            0
            >>> NONE().unwrap_or(0)
            0
        """
        return self.unwrap_or_else(lambda: default)

    def unwrap_or_else(self, callback: Callable[[], S]) -> S:
        """
        Returns the contained value or computes it from ``callback``.

        Args:
            callback: The the default callback.

        Returns:
            The contained value if the :py:class:`Option` is ``Some``,
            otherwise ``callback()``.

        Examples:
            >>> Some(0).unwrap_or_else(lambda: 111)
            0
            >>> NONE().unwrap_or_else(lambda: 'ha')
            'ha'
        """
        return self._val if self._is_some else callback()

    def map(self, callback: Callable[[S], U]) -> "Option[U]":
        """
        Applies the ``callback`` with the contained value as its argument or
        returns :py:data:`NONE`.

        Args:
            callback: The callback to apply to the contained value.

        Returns:
            The ``callback`` result wrapped in an :class:`Option` if the
            contained value is ``Some``, otherwise :py:data:`NONE`

        Examples:
            >>> Some(10).map(lambda x: x * x)
            Some(100)
            >>> NONE().map(lambda x: x * x)
            NONE
        """
        return Some(callback(self._val)) if self._is_some else NONE()

    def map_or(self, callback: Callable[[S], U], default: U) -> U:
        """
        Applies the ``callback`` to the contained value or returns ``default``.

        Args:
            callback: The callback to apply to the contained value.
            default: The default value.

        Returns:
            The ``callback`` result if the contained value is ``Some``,
            otherwise ``default``.

        Notes:
            If you wish to use the result of a function call as ``default``,
            it is recommended to use :py:meth:`map_or_else` instead.

        Examples:
            >>> Some(0).map_or(lambda x: x + 1, 1000)
            1
            >>> NONE().map_or(lambda x: x * x, 1)
            1
        """
        return callback(self._val) if self._is_some else default

    def map_or_else(self, callback: Callable[[S], U], default: Callable[[], U]) -> U:
        """
        Applies the ``callback`` to the contained value or computes a default
        with ``default``.

        Args:
            callback: The callback to apply to the contained value.
            default: The callback fot the default value.

        Returns:
            The ``callback`` result if the contained value is ``Some``,
            otherwise the result of ``default``.

        Examples:
            >>> Some(0).map_or_else(lambda x: x * x, lambda: 1)
            0
            >>> NONE().map_or_else(lambda x: x * x, lambda: 1)
            1
        """
        return callback(self._val) if self._is_some else default()

    def and_then(self, callback: "Callable[[S], Option[U]]") -> "Option[U]":
        """
        Applies the callback to the contained value if the option
        is not :py:data:`NONE`.

        This is different than :py:meth:`Option.map` because the result
        of the callback isn't wrapped in a new :py:class:`Option`

        Args:
            callback: The callback to apply to the contained value.

        Returns:
            :py:data:`NONE` if the option is :py:data:`NONE`.

            otherwise calls `callback` with the contained value and
            returns the result.

        Examples:
            >>> def square(x): return Some(x * x)
            >>> def nope(x): return NONE
            >>> Some(2).and_then(square).and_then(square)
            Some(16)
            >>> Some(2).and_then(square).and_then(nope)
            NONE
            >>> Some(2).and_then(nope).and_then(square)
            NONE
            >>> NONE().and_then(square).and_then(square)
            NONE
        """
        return callback(self._val) if self._is_some else NONE()

    def filter(self, predicate: Callable[[S], bool]) -> "Option[S]":
        """
        Returns :py:data:`NONE` if the :py:class:`Option` is :py:data:`NONE`,
        otherwise filter the contained value by ``predicate``.

        Args:
            predicate: The fitler function.

        Returns:
            :py:data:`NONE` if the contained value is :py:data:`NONE`, otherwise:
                * The option itself if the predicate returns True
                * :py:data:`NONE` if the predicate returns False

        Examples:
            >>> Some(0).filter(lambda x: x % 2 == 1)
            NONE
            >>> Some(1).filter(lambda x: x % 2 == 1)
            Some(1)
            >>> NONE().filter(lambda x: True)
            NONE
        """
        if self._is_some and predicate(self._val):
            return self
        return NONE()

    def ok_or(self, err: E) -> "Result[S, E]":
        if self._is_some:
            return Ok(self._val)

        return Err(err)

    def ok_or_else(self, err: Callable[[], E]) -> "Result[S, E]":
        if self._is_some:
            return Ok(self._val)

        return Err(err())

    def and_(self, optb: "Option[S]") -> "Option[S]":
        if self._is_some:
            return optb
        return NONE()

    def or_(self, optb: "Option[S]") -> "Option[S]":
        if self._is_some:
            return self
        return optb

    def or_else(self, f: "Callable[[], Option[S]]") -> "Option[S]":
        if self._is_some:
            return self
        return f()

    def flatten(self: "Option[Option[S]]") -> "Option[S]":
        if self._is_some:
            return self._val
        return NONE()

    def insert(self, value: S) -> S:
        self._is_some = True
        self._val = value

        return self._val

    def get_or_insert(self, value: S) -> S:
        if not self._is_some:
            self._is_some = True
            self._val = value

        return self._val

    def get_or_insert_with(self, f: Callable[[], S]) -> S:
        if not self._is_some:
            self._is_some = True
            self._val = f()

        return self._val

    def take(self) -> "Option[S]":
        if self._is_some:
            o = Some(self._val)

            self._is_some = False
            self._val = None

            return o

        return NONE()

    def to_matchable(self) -> "MatchesSome[S] | MatchesNone":
        if self._is_some:
            return MatchesSome(self._val)

        return MatchesNone()

    def __hash__(self) -> int:
        return hash((self.__class__, self._is_some, self._val))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Option)
            and self._is_some == other._is_some
            and self._val == other._val
        )

    def __ne__(self, other: object) -> bool:
        return (
            not isinstance(other, Option)
            or self._is_some != other._is_some
            or self._val != other._val
        )

    def __lt__(self: "Option[SupportsDunderLT]", other: object) -> bool:
        if isinstance(other, Option):
            if self._is_some == other._is_some:
                return self._val < other._val if self._is_some else False
            else:
                return other._is_some
        return NotImplemented

    def __le__(self: "Option[SupportsDunderLE]", other: object) -> bool:
        if isinstance(other, Option):
            if self._is_some == other._is_some:
                return self._val <= other._val if self._is_some else True
            return other._is_some
        return NotImplemented

    def __gt__(self: "Option[SupportsDunderGT]", other: object) -> bool:
        if isinstance(other, Option):
            if self._is_some == other._is_some:
                return self._val > other._val if self._is_some else False
            else:
                return self._is_some
        return NotImplemented

    def __ge__(self: "Option[SupportsDunderGE]", other: object) -> bool:
        if isinstance(other, Option):
            if self._is_some == other._is_some:
                return self._val >= other._val if self._is_some else True
            return self._is_some
        return NotImplemented

    def __repr__(self) -> str:
        return "NONE()" if self.is_none() else f"Some({self._val!r})"


def Some(val: S) -> Option[S]:
    """Shortcut function to :py:meth:`Option.Some`."""
    return Option.Some(val)


def maybe(val: Optional[S]) -> "Option[S]":
    """
    Shortcut method to return ``Some`` or :py:data:`NONE` based on ``val``.

    Args:
        val: Some value.

    Returns:
        ``Some(val)`` if the ``val`` is not None, otherwise :py:data:`NONE`.

    Examples:
        >>> maybe(0)
        Some(0)
        >>> maybe(None)
        NONE
    """
    return NONE() if val is None else Some(val)


def NONE() -> Option[Any]:
    return Option(None, False, _force=True)


@dataclass
class MatchesOk(Generic[O]):
    value: O


@dataclass
class MatchesErr(Generic[E]):
    value: E


class Result(Generic[O, E]):
    """
    :class:`Result` is a type that either success (:meth:`Result.Ok`)
    or failure (:meth:`Result.Err`).
    To create an Ok value, use :meth:`Result.Ok` or :func:`Ok`.
    To create a Err value, use :meth:`Result.Err` or :func:`Err`.
    Calling the :class:`Result` constructor directly will raise a ``TypeError``.
    Examples:
        >>> Result.Ok(1)
        Ok(1)
        >>> Result.Err('Fail!')
        Err('Fail!')
    """

    __slots__ = ("_val", "_is_ok")

    def __init__(self, val: Union[O, E], is_ok: bool, *, _force: bool = False) -> None:
        if not _force:
            raise TypeError(
                "Cannot directly initialize, "
                "please use one of the factory functions instead."
            )
        self._val = val
        self._is_ok = is_ok

    @classmethod
    def Ok(cls, val: O) -> "Result[O, Any]":
        """
        Contains the success value.
        Args:
             val: The success value.
        Returns:
             The :class:`Result` containing the success value.
        Examples:
            >>> res = Result.Ok(1)
            >>> res
            Ok(1)
            >>> res.is_ok()
            True
        """
        return cls(val, True, _force=True)

    @classmethod
    def Err(cls, err: E) -> "Result[Any, E]":
        """
        Contains the error value.
        Args:
            err: The error value.
        Returns:
            The :class:`Result` containing the error value.
        Examples:
            >>> res = Result.Err('Oh No')
            >>> res
            Err('Oh No')
            >>> res.is_err
            True
        """
        return cls(err, False, _force=True)

    def __bool__(self) -> bool:
        return self._is_ok

    def is_ok(self) -> bool:
        """
        Returns `True` if the result is :meth:`Result.Ok`.
        Examples:
            >>> Ok(1).is_ok()
            True
            >>> Err(1).is_ok()
            False
        """
        return self._is_ok

    def is_err(self) -> bool:
        """
        Returns `True` if the result is :meth:`Result.Err`.
        Examples:
            >>> Ok(1).is_err()
            False
            >>> Err(1).is_err()
            True
        """
        return not self._is_ok

    def ok(self) -> Option[O]:
        """
        Converts from :class:`Result` [O, E] to :class:`option.option_.Option` [O].
        Returns:
            :class:`Option` containing the success value if `self` is
            :meth:`Result.Ok`, otherwise :data:`option.option_.NONE`.
        Examples:
            >>> Ok(1).ok()
            Some(1)
            >>> Err(1).ok()
            NONE()
        """
        return Option.Some(self._val) if self._is_ok else NONE()  # type: ignore

    def err(self) -> Option[E]:
        """
        Converts from :class:`Result` [T, E] to :class:`option.option_.Option` [E].
        Returns:
            :class:`Option` containing the error value if `self` is
            :meth:`Result.Err`, otherwise :data:`option.option_.NONE`.
        Examples:
            >>> Ok(1).err()
            NONE()
            >>> Err(1).err()
            Some(1)
        """
        return NONE() if self._is_ok else Option.Some(self._val)  # type: ignore

    def map(self, op: Callable[[O], U]) -> "Result[U, E]":
        """
        Applies a function to the contained :meth:`Result.Ok` value.
        Args:
            op: The function to apply to the :meth:`Result.Ok` value.
        Returns:
            A :class:`Result` with its success value as the function result
            if `self` is an :meth:`Result.Ok` value, otherwise returns
            `self`.
        Examples:
            >>> Ok(1).map(lambda x: x * 2)
            Ok(2)
            >>> Err(1).map(lambda x: x * 2)
            Err(1)
        """
        return Ok(op(self._val)) if self._is_ok else self  # type: ignore

    def and_then(self, op: "Callable[[O], Result[U, E]]") -> "Result[U, E]":
        """
        Applies a function to the contained :meth:`Result.Ok` value.
        This is different than :meth:`Result.map` because the function
        result is not wrapped in a new :class:`Result`.
        Args:
            op: The function to apply to the contained :meth:`Result.Ok` value.
        Returns:
            The result of the function if `self` is an :meth:`Result.Ok` value,
             otherwise returns `self`.
        Examples:
            >>> def sq(x): return Ok(x * x)
            >>> def err(x): return Err(x)
            >>> Ok(2).and_then(sq).and_then(sq)
            Ok(16)
            >>> Ok(2).and_then(sq).and_then(err)
            Err(4)
            >>> Ok(2).and_then(err).and_then(sq)
            Err(2)
            >>> Err(3).and_then(sq).and_then(sq)
            Err(3)
        """
        return op(self._val) if self._is_ok else self  # type: ignore

    def map_err(self, op: Callable[[E], U]) -> "Result[O, U]":
        """
        Applies a function to the contained :meth:`Result.Err` value.
        Args:
            op: The function to apply to the :meth:`Result.Err` value.
        Returns:
            A :class:`Result` with its error value as the function result
            if `self` is a :meth:`Result.Err` value, otherwise returns
            `self`.
        Examples:
            >>> Ok(1).map_err(lambda x: x * 2)
            Ok(1)
            >>> Err(1).map_err(lambda x: x * 2)
            Err(2)
        """
        return self if self._is_ok else Err(op(self._val))  # type: ignore

    def unwrap(self) -> O:
        """
        Returns the success value in the :class:`Result`.
        Returns:
            The success value in the :class:`Result`.
        Raises:
            ``ValueError`` with the message provided by the error value
             if the :class:`Result` is a :meth:`Result.Err` value.
        Examples:
            >>> Ok(1).unwrap()
            1
            >>> try:
            ...     Err(1).unwrap()
            ... except ValueError as e:
            ...     print(e)
            1
        """
        if self._is_ok:
            return self._val  # type: ignore
        raise ValueError(self._val)

    def unwrap_or(self, default: O) -> O:
        """
        Returns the success value in the :class:`Result` or ``default``.
        Args:
            default: The default return value.
        Returns:
            The success value in the :class:`Result` if it is a
            :meth:`Result.Ok` value, otherwise ``default``.
        Notes:
            If you wish to use a result of a function call as the default,
            it is recommnded to use :meth:`unwrap_or_else` instead.
        Examples:
            >>> Ok(1).unwrap_or(2)
            1
            >>> Err(1).unwrap_or(2)
            2
        """
        return self._val if self._is_ok else default  # type: ignore

    def unwrap_or_else(self, op: Callable[[E], O]) -> O:
        """
        Returns the sucess value in the :class:`Result` or computes a default
        from the error value.
        Args:
            op: The function to computes default with.
        Returns:
            The success value in the :class:`Result` if it is
             a :meth:`Result.Ok` value, otherwise ``op(E)``.
        Examples:
            >>> Ok(1).unwrap_or_else(lambda e: e * 10)
            1
            >>> Err(1).unwrap_or_else(lambda e: e * 10)
            10
        """
        return self._val if self._is_ok else op(self._val)  # type: ignore

    def expect(self, msg: Any) -> O:
        """
        Returns the success value in the :class:`Result` or raises
        a ``ValueError`` with a provided message.
        Args:
            msg: The error message.
        Returns:
            The success value in the :class:`Result` if it is
            a :meth:`Result.Ok` value.
        Raises:
            ``ValueError`` with ``msg`` as the message if the
            :class:`Result` is a :meth:`Result.Err` value.
        Examples:
            >>> Ok(1).expect('no')
            1
            >>> try:
            ...     Err(1).expect('no')
            ... except ValueError as e:
            ...     print(e)
            no
        """
        if self._is_ok:
            return self._val  # type: ignore
        raise ValueError(msg)

    def unwrap_err(self) -> E:
        """
        Returns the error value in a :class:`Result`.
        Returns:
            The error value in the :class:`Result` if it is a
            :meth:`Result.Err` value.
        Raises:
            ``ValueError`` with the message provided by the success value
             if the :class:`Result` is a :meth:`Result.Ok` value.
        Examples:
            >>> try:
            ...     Ok(1).unwrap_err()
            ... except ValueError as e:
            ...     print(e)
            1
            >>> Err('Oh No').unwrap_err()
            'Oh No'
        """
        if self._is_ok:
            raise ValueError(self._val)
        return self._val  # type: ignore

    def expect_err(self, msg: Any) -> E:
        """
        Returns the error value in a :class:`Result`, or raises a
        ``ValueError`` with the provided message.
        Args:
            msg: The error message.
        Returns:
            The error value in the :class:`Result` if it is a
            :meth:`Result.Err` value.
        Raises:
            ``ValueError`` with the message provided by ``msg`` if
            the :class:`Result` is a :meth:`Result.Ok` value.
        Examples:
            >>> try:
            ...     Ok(1).expect_err('Oh No')
            ... except ValueError as e:
            ...     print(e)
            Oh No
            >>> Err(1).expect_err('Yes')
            1
        """
        if self._is_ok:
            raise ValueError(msg)
        return self._val  # type: ignore

    def transpose(self: "Result[Option[O], E]") -> "Option[Result[O, E]]":
        if self._is_ok:
            if self._val._is_some:  # pylint: disable=protected-access
                return Some(Ok(self._val._val))  # pylint: disable=protected-access
            return NONE()
        return Some(Err(self._val))

    def to_matchable(self) -> MatchesOk[O] | MatchesErr[E]:
        if self._is_ok:
            return MatchesOk(self._val)
        else:
            return MatchesErr(self._val)

    def __repr__(self) -> str:
        return f"Ok({self._val!r})" if self._is_ok else f"Err({self._val!r})"

    def __hash__(self) -> int:
        return hash((Result, self._is_ok, self._val))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Result)
            and self._is_ok == other._is_ok
            and self._val == other._val
        )

    def __ne__(self, other: object) -> bool:
        return (
            not isinstance(other, Result)
            or self._is_ok != other._is_ok
            or self._val != other._val
        )

    def __lt__(
        self: "Result[SupportsDunderLT, SupportsDunderLT]", other: object
    ) -> bool:
        if isinstance(other, Result):
            if self._is_ok == other._is_ok:
                return self._val < other._val
            return self._is_ok
        return NotImplemented

    def __le__(
        self: "Result[SupportsDunderLE, SupportsDunderLE]", other: object
    ) -> bool:
        if isinstance(other, Result):
            if self._is_ok == other._is_ok:
                return self._val <= other._val
            return self._is_ok
        return NotImplemented

    def __gt__(
        self: "Result[SupportsDunderGT, SupportsDunderGT]", other: object
    ) -> bool:
        if isinstance(other, Result):
            if self._is_ok == other._is_ok:
                return self._val > other._val
            return other._is_ok
        return NotImplemented

    def __ge__(
        self: "Result[SupportsDunderGE, SupportsDunderGE]", other: object
    ) -> bool:
        if isinstance(other, Result):
            if self._is_ok == other._is_ok:
                return self._val >= other._val
            return other._is_ok
        return NotImplemented


def Ok(val: O) -> Result[O, Any]:
    """Shortcut function for :meth:`Result.Ok`."""
    return Result.Ok(val)


def Err(err: E) -> Result[Any, E]:
    """Shortcut function for :meth:`Result.Err`."""
    return Result.Err(err)
