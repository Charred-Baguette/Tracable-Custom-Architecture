"""
NotebookLogger
==============
Drop-in replacement for RichLogger for use in Jupyter notebooks.
Uses tqdm for progress bars and plain print for console output
instead of rich, which does not render correctly in notebooks.

The .log(message, classification, Loud) signature is identical to
Logger / RichLogger so all .display() calls in SegmentHandler work
without modification.

Because this class does not expose a .console attribute, all
getattr(logger, 'console', None) checks in SegmentHandler return
None, triggering their plain-text fallbacks automatically.
"""

from Components.Logger import Logger

try:
    from tqdm.auto import tqdm as _tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False


class _TqdmProgress:
    """Context manager that mimics the rich.Progress API used by SegmentHandler.

    Supported methods:
        add_task(description, total)  -> task_id
        update(task_id, advance, description)
    """

    def __init__(self, transient=False, disable=False):
        self._transient = transient
        self._disable   = disable
        self._bars      = {}
        self._next_id   = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        for bar in self._bars.values():
            if bar is not None:
                bar.close()
        self._bars.clear()

    def add_task(self, description='', total=100, **kwargs):
        task_id = self._next_id
        self._next_id += 1
        if _TQDM_AVAILABLE:
            bar = _tqdm(
                total=total,
                desc=description,
                disable=self._disable,
                leave=not self._transient,
            )
        else:
            bar = None
        self._bars[task_id] = bar
        return task_id

    def update(self, task_id, advance=0, description=None, **kwargs):
        bar = self._bars.get(task_id)
        if bar is None:
            return
        if description is not None:
            bar.set_description(description)
        if advance:
            bar.update(advance)


class NotebookLogger(Logger):
    """Logger for Jupyter notebooks.

    Inherits all file-logging and deduplication logic from Logger.
    Adds tqdm-backed progress bars via make_progress(), matching the
    API that SegmentHandler expects from RichLogger.

    No .console attribute is exposed, so SegmentHandler's rich table
    rendering paths are bypassed and their plain-text fallbacks are
    used instead.
    """

    def __init__(self, filename: str, log_level: int) -> None:
        super().__init__(filename, log_level)

    def make_progress(self, transient: bool = False, disable: bool = False) -> _TqdmProgress:
        """Return a tqdm-backed progress context manager.

        Usage mirrors rich.Progress:

            with logger.make_progress() as progress:
                task = progress.add_task("Doing work", total=n)
                for item in items:
                    ...
                    progress.update(task, advance=1)
        """
        return _TqdmProgress(transient=transient, disable=disable)
