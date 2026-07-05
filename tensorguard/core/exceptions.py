from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console(stderr=True)

class ShapeMismatchError(Exception):
    """
    Exception raised for errors in tensor shape matching, with rich terminal formatting.
    """
    def __init__(
        self, 
        tensor_name: str, 
        expected_shape: str, 
        actual_shape: tuple, 
        error_msg: str
    ):
        self.tensor_name = tensor_name
        self.expected_shape = expected_shape
        self.actual_shape = actual_shape
        self.error_msg = error_msg
        
        # We print the rich panel immediately before the standard traceback kills the process
        self._render_error()
        super().__init__(self.error_msg)
        
    def _render_error(self) -> None:
        table = Table(show_header=False, box=None)
        table.add_column("Property", style="bold")
        table.add_column("Value")
        
        table.add_row("Tensor Name:", f"[cyan]{self.tensor_name}[/cyan]")
        table.add_row("Expected Shape:", f"[green]{self.expected_shape}[/green]")
        table.add_row("Actual Shape:", f"[red]{self.actual_shape}[/red]")
        table.add_row("Details:", f"[yellow]{self.error_msg}[/yellow]")
        
        panel = Panel(
            table, 
            title="[bold red]❌ Tensor Shape Mismatch[/bold red]", 
            expand=False, 
            border_style="red"
        )
        
        console.print(panel)
