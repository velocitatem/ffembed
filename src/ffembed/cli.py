"""ffembed: a tiny local semantic index for your files.

    ffembed watch ~/notes --filter "*.md"
    ffembed start
    ffembed search "that thing about debounce"
"""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from . import db
from .embed import DEFAULT_MODEL, GARDEN
from .indexer import index_target, remove_missing_files
from .vision import DEFAULT_VISION_MODEL, VISION_GARDEN

console = Console()


@click.group()
def main():
    """ffembed - a tiny local semantic index for your files."""


@main.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option("--filter", "pattern", default="*.md", show_default=True, help="Glob pattern for filenames to index.")
@click.option("--model", default=DEFAULT_MODEL, show_default=True, help=f"Text embedding model. One of: {', '.join(GARDEN)} (or any fastembed model name).")
@click.option("--vision-model", default=DEFAULT_VISION_MODEL, show_default=True, help=f"Image embedding model. One of: {', '.join(VISION_GARDEN)} (or any DINOv3 HF name).")
def watch(directory: str, pattern: str, model: str, vision_model: str):
    """Register DIRECTORY for indexing and index it once immediately.

    Image files (jpg/png/webp/gif/bmp/tiff) are embedded with the DINOv3
    vision model. Text files are embedded with the text model.
    """
    from . import daemon

    root = str(Path(directory).resolve())
    with db.cursor() as conn:
        target_id = db.add_target(conn, root, pattern, model, vision_model=vision_model)
        target_row = conn.execute("SELECT * FROM targets WHERE id = ?", (target_id,)).fetchone()
        with console.status(f"Indexing {root} ({pattern})..."):
            count = index_target(conn, target_row)
    console.print(f"[green]Watching[/green] {root} [{pattern}] with text model '{model}' and vision model '{vision_model}' - indexed {count} file(s).")
    if daemon.is_running():
        console.print("[yellow]Daemon is running - restart it to pick up the new target:[/yellow] ffembed restart")
    else:
        console.print("Start the daemon to keep it in sync: [bold]ffembed start[/bold]")


@main.command()
@click.argument("directory", type=click.Path(file_okay=False))
def unwatch(directory: str):
    """Stop watching DIRECTORY and drop its indexed data."""
    from . import daemon

    root = str(Path(directory).resolve())
    with db.cursor() as conn:
        removed = db.remove_target(conn, root)
    if removed:
        console.print(f"[green]Removed[/green] {root}")
        if daemon.is_running():
            console.print("[yellow]Restart the daemon to stop watching it:[/yellow] ffembed restart")
    else:
        console.print(f"[red]Not watching[/red] {root}")


@main.command("list")
def list_targets():
    """Show watched directories and index stats."""
    with db.cursor() as conn:
        targets = db.list_targets(conn)
        table = Table(title="ffembed targets")
        table.add_column("path")
        table.add_column("filter")
        table.add_column("model")
        table.add_column("vision")
        table.add_column("files")
        table.add_column("chunks")
        for t in targets:
            files = conn.execute("SELECT COUNT(*) c FROM files WHERE target_id = ?", (t["id"],)).fetchone()["c"]
            chunks = conn.execute(
                "SELECT COUNT(*) c FROM chunks JOIN files ON files.id = chunks.file_id WHERE files.target_id = ?",
                (t["id"],),
            ).fetchone()["c"]
            vision = t["vision_model"]
            table.add_row(t["path"], t["pattern"], t["model"], vision if vision else DEFAULT_VISION_MODEL, str(files), str(chunks))
    console.print(table)


@main.command()
@click.argument("directory", required=False, type=click.Path(exists=True, file_okay=False))
def reindex(directory: str | None):
    """Force a full reindex of one target, or all targets if none given."""
    root = str(Path(directory).resolve()) if directory else None
    with db.cursor() as conn:
        targets = db.list_targets(conn)
        if root:
            targets = [t for t in targets if t["path"] == root]
            if not targets:
                console.print(f"[red]Not watching[/red] {root}")
                return
        for t in targets:
            with console.status(f"Reindexing {t['path']}..."):
                count = index_target(conn, t, force=True)
                removed = remove_missing_files(conn, t)
            console.print(f"[green]{t['path']}[/green]: reindexed {count}, removed {removed} stale file(s).")


@main.command()
@click.argument("query")
@click.option("--dir", "target_dir", default=None, help="Restrict search to one watched directory.")
@click.option("-k", "top_k", default=5, show_default=True, help="Number of results to return.")
def search(query: str, target_dir: str | None, top_k: int):
    """Semantic search across indexed content."""
    from .search import search as run_search

    target_path = str(Path(target_dir).resolve()) if target_dir else None
    with db.cursor() as conn:
        with console.status("Searching..."):
            results = run_search(conn, query, target_path=target_path, k=top_k)
    if not results:
        console.print("[yellow]No results.[/yellow] Have you run 'ffembed watch <dir>' yet?")
        return
    for score, row in results:
        if row["kind"] == "image":
            snippet = "[image]"
        else:
            snippet = row["text"].strip().replace("\n", " ")
            if len(snippet) > 220:
                snippet = snippet[:220] + "…"
        console.print(f"[bold cyan]{score:.3f}[/bold cyan]  [dim]{row['file_path']}[/dim]")
        console.print(f"    {snippet}\n")


@main.command()
@click.option("--debounce", default=2.0, show_default=True, help="Seconds of quiet before a changed file is re-embedded.")
def start(debounce: float):
    """Start the background watch daemon."""
    from . import daemon

    pid = daemon.start(debounce_seconds=debounce)
    console.print(f"[green]ffembed daemon running[/green] (pid {pid})")


@main.command()
def stop():
    """Stop the background watch daemon."""
    from . import daemon

    if daemon.stop():
        console.print("[green]Stopped.[/green]")
    else:
        console.print("[yellow]Daemon was not running.[/yellow]")


@main.command()
@click.option("--debounce", default=2.0, show_default=True)
def restart(debounce: float):
    """Restart the background watch daemon."""
    from . import daemon

    daemon.stop()
    pid = daemon.start(debounce_seconds=debounce)
    console.print(f"[green]ffembed daemon running[/green] (pid {pid})")


@main.command()
def status():
    """Show daemon status and index stats."""
    from . import daemon

    pid = daemon.is_running()
    if pid:
        console.print(f"[green]daemon running[/green] (pid {pid})")
    else:
        console.print("[yellow]daemon not running[/yellow]")
    with db.cursor() as conn:
        s = db.stats(conn)
    console.print(f"{s['targets']} target(s), {s['files']} file(s), {s['chunks']} chunk(s)")


@main.command()
def models():
    """List the text and vision embedding gardens."""
    text_table = Table(title="text embedding garden")
    text_table.add_column("alias")
    text_table.add_column("model")
    for alias, name in GARDEN.items():
        marker = " (default)" if alias == DEFAULT_MODEL else ""
        text_table.add_row(alias + marker, name)
    console.print(text_table)

    vision_table = Table(title="vision embedding garden")
    vision_table.add_column("alias")
    vision_table.add_column("model")
    for alias, name in VISION_GARDEN.items():
        marker = " (default)" if alias == DEFAULT_VISION_MODEL else ""
        vision_table.add_row(alias + marker, name)
    console.print(vision_table)


if __name__ == "__main__":
    main()
