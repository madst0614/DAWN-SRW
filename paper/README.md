# DAWN-SRW paper

The version-controlled paper source and the latest built PDF live in this
directory. `main.pdf` is committed so the current draft can be opened directly
from GitHub. LaTeX intermediate files are ignored.

## Layout

- `main.tex`: paper entry point
- `main.pdf`: latest built draft, tracked by Git
- `references.bib`: BibTeX database
- `figures/`: figures referenced by the paper
- `icml2026.*`, `fancyhdr.sty`: venue-provided style files
- `imports/dawn_srw-original.zip`: preserved local import archive (ignored by Git)

## Edit with live PDF preview

Install [MiKTeX](https://miktex.org/howto/install-miktex) for the current user,
enable automatic installation of missing packages, and install the recommended
VS Code extension when prompted.

Open `paper/main.tex` in VS Code and open the LaTeX Workshop PDF viewer once.
The repository settings rebuild `paper/main.pdf` whenever a `.tex` file is
saved, and the PDF tab refreshes automatically.

The build uses MiKTeX's `texify` driver, which runs pdfLaTeX and BibTeX as many
times as needed to resolve citations and cross-references. It does not require a
separate Perl installation.

For a manual build:

```powershell
Set-Location paper
texify --pdf --max-iterations=5 --tex-option=--synctex=1 --tex-option=--halt-on-error main.tex
```

Commit `main.tex`, related figures or bibliography changes, and the refreshed
`main.pdf` together.
