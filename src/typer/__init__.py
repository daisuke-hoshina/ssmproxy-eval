"""A lightweight stand-in for :mod:`typer` used in tests."""

from __future__ import annotations

from typing import Any, Callable, Optional


def echo(message: str) -> None:  # pragma: no cover - trivial wrapper
    print(message)


def Option(default: Any, *_, help: Optional[str] = None, **__) -> Any:
    return default


def Argument(default: Any, *_, **__) -> Any:
    return default


class Typer:
    def __init__(self, *_, **__):
        self._commands: list[Callable[..., Any]] = []

    def command(self, *_, **__) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self._commands.append(func)
            return func

        return decorator

    def add_typer(self, *_args, **__kwargs) -> None:  # pragma: no cover - no-op
        return None

    def __call__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - CLI shim
        for command in self._commands:
            command(*args, **kwargs)
