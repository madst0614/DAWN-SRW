# DAWN-SRW paper

The paper is organized by release version:

```text
paper/
├── v9/
│   ├── source/
│   └── DAWN-SRW-v9.pdf
├── v10/
│   ├── source/
│   └── DAWN-SRW-v10.pdf
├── v11/
│   ├── source/
│   └── DAWN-SRW-v11.pdf
├── v12/
│   ├── source/
│   └── DAWN-SRW-v12.pdf
└── v13/
    ├── source/
    └── DAWN-SRW-v13.pdf
```

- `v9` is the Zenodo-published snapshot and is preserved by the `paper-v9`
  Git tag.
- `v10` is the preserved prior working draft.
- `v11` is the preserved prior working draft.
- `v12` is the preserved prior working draft.
- `v13` is the active working draft. It is not tagged until release.
- Version numbers are repository metadata only; they are not printed on the
  paper's first page.

## Edit and build

Open `paper/v13/source/main.tex` in VS Code. With the recommended LaTeX
Workshop extension installed, saving a TeX file runs `build.ps1` automatically.
The script builds inside `source/` and refreshes the PDF directly at:

```text
paper/v13/DAWN-SRW-v13.pdf
```

To build manually from PowerShell:

```powershell
cd paper\v13\source
.\build.ps1
```

Commit the changed source files and the refreshed versioned PDF together.
