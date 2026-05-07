"""CLI script for bayes_climsim_eval."""

import typer
from rich.console import Console

app = typer.Typer()
console = Console()


@app.command()
def main():
    """CLI script for bayes_climsim_eval."""
    console.print(r"""
  $$\      $$\           $$\                                             
  $$ | $\  $$ |          $$ |                                            
  $$ |$$$\ $$ | $$$$$$\  $$ | $$$$$$$\  $$$$$$\  $$$$$$\$$$$\   $$$$$$\  
  $$ $$ $$\$$ |$$  __$$\ $$ |$$  _____|$$  __$$\ $$  _$$  _$$\ $$  __$$\ 
  $$$$  _$$$$ |$$$$$$$$ |$$ |$$ /      $$ /  $$ |$$ / $$ / $$ |$$$$$$$$ |
  $$$  / \$$$ |$$   ____|$$ |$$ |      $$ |  $$ |$$ | $$ | $$ |$$   ____|
  $$  /   \$$ |\$$$$$$$\ $$ |\$$$$$$$\ \$$$$$$  |$$ | $$ | $$ |\$$$$$$$\ 
  \__/     \__| \_______|\__| \_______| \______/ \__| \__| \__| \_______|
                                                                      
    """, style="green")
    # https://patorjk.com/software/taag/
    console.print("Replace this message by putting your code into "
               "bayes_climsim_eval.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")


if __name__ == "__main__":
    app()
