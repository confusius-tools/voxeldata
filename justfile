set windows-shell := ["pwsh.exe", "-c"]

# Print the help message.
@help:
    echo "Usage: just [RECIPE]\n"
    just --list

# Build documentation.
docs:
    uv run zensical build --strict

# Serve documentation locally for development.
serve-docs:
    uv run zensical serve

# Clean documentation build artifacts.
[unix]
clean-docs:
    rm -rf site/

# Clean documentation build artifacts.
[windows]
clean-docs:
    if (Test-Path site) { Remove-Item -Recurse -Force site }

# Run all tests.
test:
    uv run pytest

# Run tests with verbose output.
test-verbose:
    uv run pytest -v

# Run all pre-commit hooks.
pre-commit:
    uv run prek run --all-files

# Aliases
alias d := docs
alias cd := clean-docs
alias sd := serve-docs
alias t := test
alias tv := test-verbose
alias pc := pre-commit
