"""Utilities for invoking async tools from synchronous agent paths."""

import asyncio
import atexit
import concurrent.futures
import functools
import logging
import threading
from collections.abc import Callable
from typing import Any, get_type_hints

from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)

# Shared thread pool for sync tool invocation in async environments.
_SYNC_TOOL_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=10, thread_name_prefix="tool-sync")

atexit.register(lambda: _SYNC_TOOL_EXECUTOR.shutdown(wait=False))

# Persistent event loop running in a dedicated daemon thread.
# This ensures that MCP session pool entries are always created in the same
# event loop and can be reused across tool calls without being evicted due to
# loop mismatch (which causes repeated browser open/close with Playwright).
_PERSISTENT_LOOP: asyncio.AbstractEventLoop | None = None
_PERSISTENT_LOOP_LOCK = threading.Lock()


def _get_persistent_loop() -> asyncio.AbstractEventLoop:
    """Return (and lazily create) a persistent event loop for sync tool calls."""
    global _PERSISTENT_LOOP
    if _PERSISTENT_LOOP is None or _PERSISTENT_LOOP.is_closed():
        with _PERSISTENT_LOOP_LOCK:
            if _PERSISTENT_LOOP is None or _PERSISTENT_LOOP.is_closed():
                _PERSISTENT_LOOP = asyncio.new_event_loop()
                thread = threading.Thread(
                    target=_PERSISTENT_LOOP.run_forever,
                    name="tool-sync-loop",
                    daemon=True,
                )
                thread.start()
    return _PERSISTENT_LOOP


def _shutdown_persistent_loop() -> None:
    """Stop the persistent loop on interpreter exit."""
    if _PERSISTENT_LOOP is not None and not _PERSISTENT_LOOP.is_closed():
        _PERSISTENT_LOOP.call_soon_threadsafe(_PERSISTENT_LOOP.stop)


atexit.register(_shutdown_persistent_loop)


def _get_runnable_config_param(func: Callable[..., Any]) -> str | None:
    """Return the coroutine parameter that expects LangChain RunnableConfig."""
    if isinstance(func, functools.partial):
        func = func.func

    try:
        type_hints = get_type_hints(func)
    except Exception:
        return None

    for name, type_ in type_hints.items():
        if type_ is RunnableConfig:
            return name
    return None


def make_sync_tool_wrapper(coro: Callable[..., Any], tool_name: str) -> Callable[..., Any]:
    """Build a synchronous wrapper for an asynchronous tool coroutine.

    Args:
        coro: Async callable backing a LangChain tool.
        tool_name: Tool name used in error logs.

    Returns:
        A sync callable suitable for ``BaseTool.func``.

    Notes:
        If ``coro`` declares a ``RunnableConfig`` parameter, this wrapper
        exposes ``config: RunnableConfig`` so LangChain can inject runtime
        config and then forwards it to the coroutine's detected config
        parameter. This covers DeerFlow's current config-sensitive tools, such
        as ``invoke_acp_agent``.

        This wrapper intentionally does not synthesize a dynamic function
        signature. A future async tool with a normal user-facing argument named
        ``config`` and a separate ``RunnableConfig`` parameter named something
        else, such as ``run_config``, may collide with LangChain's injected
        ``config`` argument. Rename that user-facing field or extend this
        helper before using that signature.
    """
    config_param = _get_runnable_config_param(coro)

    def run_coroutine(*args: Any, **kwargs: Any) -> Any:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        try:
            if loop is not None and loop.is_running():
                # We're inside an async context (e.g. LangGraph). Schedule on the
                # persistent loop so MCP session pool entries remain valid across calls.
                persistent_loop = _get_persistent_loop()
                future = asyncio.run_coroutine_threadsafe(coro(*args, **kwargs), persistent_loop)
                return future.result()
            # No running loop – use the persistent loop as well to keep session
            # pool entries stable.
            persistent_loop = _get_persistent_loop()
            future = asyncio.run_coroutine_threadsafe(coro(*args, **kwargs), persistent_loop)
            return future.result()
        except Exception as e:
            logger.error("Error invoking tool %r via sync wrapper: %s", tool_name, e, exc_info=True)
            raise

    if config_param:

        def sync_wrapper(*args: Any, config: RunnableConfig = None, **kwargs: Any) -> Any:
            if config is not None or config_param not in kwargs:
                kwargs[config_param] = config
            return run_coroutine(*args, **kwargs)

        return sync_wrapper

    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        return run_coroutine(*args, **kwargs)

    return sync_wrapper
