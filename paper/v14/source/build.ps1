$ErrorActionPreference = "Stop"

$sourceDir = $PSScriptRoot
$versionDir = Split-Path -Parent $sourceDir
$versionName = Split-Path -Leaf $versionDir
$versionedPdf = Join-Path $versionDir "DAWN-SRW-$versionName.pdf"
$texifyCommand = Get-Command texify -ErrorAction SilentlyContinue

if ($null -ne $texifyCommand) {
    $texifyPath = $texifyCommand.Source
} else {
    $texifyCandidates = @(
        (Join-Path $env:LOCALAPPDATA "Programs\MiKTeX\miktex\bin\x64\texify.exe"),
        (Join-Path $env:ProgramFiles "MiKTeX\miktex\bin\x64\texify.exe")
    )
    $texifyPath = $texifyCandidates |
        Where-Object { Test-Path -LiteralPath $_ } |
        Select-Object -First 1
}

if (-not $texifyPath) {
    throw "MiKTeX texify was not found. Install MiKTeX or add it to PATH."
}

Push-Location $sourceDir
try {
    & $texifyPath `
        --pdf `
        --max-iterations=5 `
        --tex-option=--synctex=1 `
        --tex-option=--halt-on-error `
        main.tex

    if ($LASTEXITCODE -ne 0) {
        throw "LaTeX build failed with exit code $LASTEXITCODE."
    }

    Copy-Item -LiteralPath (Join-Path $sourceDir "main.pdf") `
        -Destination $versionedPdf `
        -Force
} finally {
    Pop-Location
}
