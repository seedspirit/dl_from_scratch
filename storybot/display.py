"""공유 콘솔과 단계별 진행 표시."""

from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

console = Console(stderr=True)


def byte_progress(*, enabled: bool) -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TextColumn("경과"),
        TimeElapsedColumn(),
        TextColumn("남음"),
        TimeRemainingColumn(),
        console=console,
        disable=not enabled,
    )


def step_progress(*, enabled: bool) -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TextColumn("경과"),
        TimeElapsedColumn(),
        TextColumn("남음"),
        TimeRemainingColumn(),
        console=console,
        disable=not enabled,
    )
