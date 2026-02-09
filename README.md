# Cognexia
A CLI-Based Cross-Platform Local Knowledge Base System Using NLP and Intent-Aware Inference over Heterogeneous Local Data

## CLI

This project exposes a minimal, stable CLI contract that will later plug into
NLP, retrieval, and inference components. For now, it accepts a user query and
returns a static response. Command routing lives in a dedicated layer to keep
the entry point minimal and decoupled from execution logic.

### Usage

Run the CLI module directly:

- `python -m cognexia.cli.main "Summarize my project files"`
- `python -m cognexia.cli.main query "Summarize my project files"`
- `python -m cognexia.cli.main version`

Expected output:

```
[Cognexia] Inference engine not implemented yet.
```

### Help

```
python -m cognexia.cli.main --help
```

### Notes

- Running with no input displays the help text.
- The CLI contract is defined in `handle_query(query: str) -> str`.
- The query handler lives in `cognexia.inference.engine`.
- Command routing is implemented in `cognexia.cli.commands`.

## Getting started (open-source developers)

### Prerequisites

- Python 3.11+ (3.13 supported)
- Git

### Setup

Clone the repo and create a virtual environment:

```
git clone https://github.com/<your-org-or-user>/Cognexia.git
cd Cognexia
python -m venv .venv
```

Activate the environment:

- Windows (PowerShell):
```
.\.venv\Scripts\Activate.ps1
```

- macOS/Linux (bash/zsh):
```
source .venv/bin/activate
```

### Project layout

```
src/
	cognexia/
		cli/
			main.py
			commands.py
		inference/
			engine.py
```

### Run locally

```
python -m cognexia.cli.main "Summarize my project files"
```

### Tests

There are no automated tests yet. When tests are added, they should be runnable
via standard Python test discovery:

```
python -m unittest discover -s . -p "test*.py"
```

## Contributing

Contributions are welcome. Please keep changes focused and scoped to a single
concern. If you plan a large change, open an issue first to discuss the design.

### Development guidelines

- Keep the CLI contract stable in `handle_query(query: str) -> str`.
- Avoid adding NLP, retrieval, or inference logic in the CLI layer.
- Prefer small, well-documented changes.

## License
Add a license file when ready to open-source this repository.
