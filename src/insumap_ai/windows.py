"""Fixed-size windows over a sequence."""

from __future__ import annotations

from collections.abc import Iterator, Sequence


class SlidingWindowIterator[T]:
    """Yields every contiguous window of ``size`` items, oldest first.

    Pattern: P14 Iterator
    """

    def __init__(self, items: Sequence[T], size: int, step: int = 1) -> None:
        if size < 1 or step < 1:
            raise ValueError("size and step must be positive")
        self._items = items
        self._size = size
        self._step = step
        self._start = 0

    def __iter__(self) -> Iterator[tuple[T, ...]]:
        return self

    def __next__(self) -> tuple[T, ...]:
        end = self._start + self._size
        if end > len(self._items):
            raise StopIteration
        window = tuple(self._items[self._start : end])
        self._start += self._step
        return window
