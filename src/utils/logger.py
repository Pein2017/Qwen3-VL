"""
Unified, rank-aware logging for Qwen3-VL (src namespace only).

Goals:
- Single logger hierarchy rooted at `src`.
- Rank filtering: log from rank 0 by default; opt-in all-rank via QWEN3VL_VERBOSE=1.
- Debug flag should raise verbosity only for our code; do not touch torch/transformers/cuda.
- No root/global logging mutations that could leak to third-party libraries.
"""

import logging
import os
from typing_extensions import override

BASE_LOGGER_NAME = "src"

# State
_global_debug_flag: bool = False
_global_log_level: int = logging.INFO
_logging_configured: bool = False


def get_rank() -> int:
    """
    Get the current process rank in distributed training.

    Returns:
        Process rank (0 for single GPU or main process)
    """
    # Check various distributed training environment variables
    rank = os.environ.get("RANK")
    if rank is not None:
        return int(rank)

    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        return int(local_rank)

    # Not in distributed mode
    return 0


def is_main_process() -> bool:
    """
    Check if current process is the main process (rank 0).

    Returns:
        True if main process or not in distributed mode
    """
    return get_rank() == 0


def should_log() -> bool:
    """
    Determine if current process should log messages.

    Returns:
        True if rank 0 or verbose mode enabled
    """
    # Check if verbose mode is enabled
    verbose = os.environ.get("QWEN3VL_VERBOSE", "0").strip().lower()
    if verbose in ("1", "true", "yes"):
        return True

    # Only log from rank 0
    return is_main_process()


class RankFilter(logging.Filter):
    """
    Logging filter that blocks messages from non-main processes.

    In distributed training, this ensures only rank 0 logs by default.
    Can be overridden with QWEN3VL_VERBOSE=1 environment variable.
    """

    @override
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter log records based on process rank.

        Args:
            record: Log record to filter

        Returns:
            True if record should be logged
        """
        return should_log()


def is_verbose_enabled() -> bool:
    verbose = os.environ.get("QWEN3VL_VERBOSE", "0").strip().lower()
    return verbose in ("1", "true", "yes", "y")


def _base_logger() -> logging.Logger:
    return logging.getLogger(BASE_LOGGER_NAME)


def _ensure_base_logger() -> None:
    """Install handler/formatter/rank filter on the src root logger once."""
    global _logging_configured
    if _logging_configured:
        return

    base = _base_logger()
    if not base.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        handler.addFilter(RankFilter())
        base.addHandler(handler)
    base.propagate = False  # contain logs within src namespace
    _logging_configured = True


def _logger_name(name: str | None) -> str:
    if name in (None, "", BASE_LOGGER_NAME):
        return BASE_LOGGER_NAME
    return f"{BASE_LOGGER_NAME}.{name}"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger under the `src` namespace."""
    _ensure_base_logger()
    logger = logging.getLogger(_logger_name(name))
    # Child loggers inherit handlers/filters/levels from base when level=NOTSET
    if logger.level == logging.NOTSET:
        logger.setLevel(logging.NOTSET)
    logger.propagate = True
    return logger


def configure_logging(
    *, level: int = logging.INFO, debug: bool = False, verbose: bool = False
) -> None:
    """
    Configure the src logging hierarchy.

    - Affects only loggers under the `src` namespace.
    - Respects rank filtering unless `verbose` is True or QWEN3VL_VERBOSE=1.
    - When debug=True, elevates to DEBUG for src loggers only.
    """
    global _global_debug_flag, _global_log_level

    if verbose:
        os.environ["QWEN3VL_VERBOSE"] = "1"
    elif not is_verbose_enabled():
        os.environ["QWEN3VL_VERBOSE"] = "0"

    _global_debug_flag = bool(debug)
    resolved_level = logging.DEBUG if debug else int(level)
    _global_log_level = resolved_level

    _ensure_base_logger()
    base = _base_logger()
    base.setLevel(resolved_level)
    base.propagate = False

    # Normalize child loggers under src.* to inherit from base
    for name, logger_obj in logging.Logger.manager.loggerDict.items():
        if not isinstance(logger_obj, logging.Logger):
            continue
        if name == BASE_LOGGER_NAME or name.startswith(f"{BASE_LOGGER_NAME}."):
            logger_obj.setLevel(logging.NOTSET)
            logger_obj.propagate = True

    # Ensure formatter/filter present on base handlers
    for handler in base.handlers:
        handler.setLevel(resolved_level)
        has_rank_filter = any(isinstance(f, RankFilter) for f in handler.filters)
        if not has_rank_filter:
            handler.addFilter(RankFilter())


def set_global_debug(debug: bool = True) -> None:
    configure_logging(
        level=_global_log_level, debug=debug, verbose=is_verbose_enabled()
    )


def is_global_debug_enabled() -> bool:
    return _global_debug_flag


def set_log_level(level: int) -> None:
    configure_logging(level=level, debug=False, verbose=is_verbose_enabled())


def enable_verbose_logging() -> None:
    os.environ["QWEN3VL_VERBOSE"] = "1"
    configure_logging(level=_global_log_level, debug=_global_debug_flag, verbose=True)


def disable_verbose_logging() -> None:
    os.environ["QWEN3VL_VERBOSE"] = "0"
    configure_logging(level=_global_log_level, debug=_global_debug_flag, verbose=False)


# Convenience logger for this module
logger = get_logger(__name__)
