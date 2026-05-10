#!/usr/bin/env pwsh
# Usage:
#   .\scripts\screenshot.ps1 <viewer.html> <out.png> [width] [height]
#   .\scripts\screenshot.ps1 <viewer.html> <out.png> [width] [height] --multi
#
# Headlessly renders the viewer into a PNG on Windows. With --multi, captures
# 7 camera angles (iso/front/back/left/right/top/bottom) via the viewer's
# ?angle=... URL param and composes them into a 2x4 montage using .NET drawing
# APIs, so ImageMagick is not required.

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

function Show-Usage {
    Write-Host "usage: .\scripts\screenshot.ps1 <viewer.html> <out.png> [width] [height] [--multi] [--highlight <id[,id...]>] [--solo]"
}

function Format-FileSize([long]$Bytes) {
    if ($Bytes -ge 1GB) { return "{0:N1}G" -f ($Bytes / 1GB) }
    if ($Bytes -ge 1MB) { return "{0:N1}M" -f ($Bytes / 1MB) }
    if ($Bytes -ge 1KB) { return "{0:N1}K" -f ($Bytes / 1KB) }
    return "$Bytes B"
}

function Get-BrowserPath {
    $candidates = New-Object System.Collections.Generic.List[string]

    function Add-CandidatePath([string]$Base, [string]$Child) {
        if ($Base) {
            $candidates.Add((Join-Path $Base $Child))
        }
    }

    foreach ($envName in @("CHROME", "CHROME_BIN", "EDGE", "EDGE_BIN")) {
        $value = [Environment]::GetEnvironmentVariable($envName)
        if ($value) {
            $candidates.Add($value)
        }
    }

    Add-CandidatePath $env:ProgramFiles "Google\Chrome\Application\chrome.exe"
    Add-CandidatePath ${env:ProgramFiles(x86)} "Google\Chrome\Application\chrome.exe"
    Add-CandidatePath $env:LocalAppData "Google\Chrome\Application\chrome.exe"
    Add-CandidatePath $env:ProgramFiles "Microsoft\Edge\Application\msedge.exe"
    Add-CandidatePath ${env:ProgramFiles(x86)} "Microsoft\Edge\Application\msedge.exe"
    Add-CandidatePath $env:LocalAppData "Microsoft\Edge\Application\msedge.exe"

    foreach ($cmdName in @("chrome.exe", "msedge.exe")) {
        $cmd = Get-Command $cmdName -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($cmd -and $cmd.Source) {
            $candidates.Add($cmd.Source)
        }
    }

    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            return (Get-Item -LiteralPath $candidate).FullName
        }
    }

    throw "Chrome or Edge was not found. Install one, add it to PATH, or set CHROME/CHROME_BIN/EDGE/EDGE_BIN."
}

function Assert-NonEmptyFile([string]$Path, [string]$Message, [int]$TimeoutMs = 10000) {
    $deadline = [DateTime]::UtcNow.AddMilliseconds($TimeoutMs)
    do {
        if (Test-Path -LiteralPath $Path -PathType Leaf) {
            $item = Get-Item -LiteralPath $Path
            if ($item.Length -gt 0) {
                return
            }
        }
        Start-Sleep -Milliseconds 100
    } while ([DateTime]::UtcNow -lt $deadline)

    if (Test-Path -LiteralPath $Path -PathType Leaf) {
        $item = Get-Item -LiteralPath $Path
        throw "$Message ($Path is empty, length $($item.Length))"
    }
    throw "$Message ($Path was not created)"
}

function Ensure-ParentDirectory([string]$Path) {
    $parent = [System.IO.Path]::GetDirectoryName([System.IO.Path]::GetFullPath($Path))
    if ($parent -and !(Test-Path -LiteralPath $parent -PathType Container)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
}

function Invoke-Shot([string]$Angle, [string]$OutputPath) {
    Ensure-ParentDirectory $OutputPath

    $query = New-Object System.Collections.Generic.List[string]
    if ($Angle) {
        $query.Add("angle=$([System.Uri]::EscapeDataString($Angle))")
    }
    $query.Add("ui=0")
    if ($script:Highlight) {
        $query.Add("highlight=$([System.Uri]::EscapeDataString($script:Highlight))")
    }
    if ($script:SoloHighlight) {
        $query.Add("solo=1")
    }
    $url = "${script:ViewerUrl}?$($query -join '&')"

    $log = Join-Path ([System.IO.Path]::GetTempPath()) "cpd-shot.log"
    $browserArgs = @(
        "--headless=new",
        "--hide-scrollbars",
        "--enable-webgl",
        "--use-angle=d3d11",
        "--virtual-time-budget=8000",
        "--window-size=$script:Width,$script:Height",
        "--screenshot=$OutputPath",
        $url
    )

    for ($attempt = 1; $attempt -le 3; $attempt++) {
        if (Test-Path -LiteralPath $OutputPath -PathType Leaf) {
            Remove-Item -LiteralPath $OutputPath -Force
        }

        Set-Variable -Name LASTEXITCODE -Value 0 -Scope Global
        & $script:Browser @browserArgs *> $log
        $exitCode = (Get-Variable -Name LASTEXITCODE -Scope Global).Value
        if ($exitCode -ne 0) {
            if ($attempt -eq 3) {
                throw "browser screenshot failed with exit code $exitCode (see $log)"
            }
            Start-Sleep -Milliseconds (250 * $attempt)
            continue
        }

        try {
            Assert-NonEmptyFile $OutputPath "screenshot failed" 1500
            return
        }
        catch {
            if ($attempt -eq 3) {
                throw
            }
            Start-Sleep -Milliseconds (250 * $attempt)
        }
    }
}

function Ensure-DrawingAssembly {
    if ($script:DrawingLoaded) {
        return
    }
    Add-Type -AssemblyName System.Drawing
    $script:DrawingLoaded = $true
}

function Add-ImageLabel([string]$Path, [string]$Text) {
    Ensure-DrawingAssembly

    $src = [System.Drawing.Image]::FromFile($Path)
    $bmp = $null
    $g = $null
    $font = $null
    $bg = $null
    $fg = $null
    try {
        $bmp = New-Object System.Drawing.Bitmap $src.Width, $src.Height
        $g = [System.Drawing.Graphics]::FromImage($bmp)
        $g.DrawImage($src, 0, 0, $src.Width, $src.Height)

        $fontSize = [Math]::Max(14, [Math]::Round($src.Height / 34))
        $font = New-Object System.Drawing.Font "Arial", $fontSize, ([System.Drawing.FontStyle]::Bold), ([System.Drawing.GraphicsUnit]::Pixel)
        $label = " $Text "
        $size = $g.MeasureString($label, $font)
        $pad = [Math]::Max(10, [Math]::Round($src.Height / 54))
        $x = [single]($src.Width - $size.Width - $pad)
        $y = [single]$pad
        $rect = New-Object System.Drawing.RectangleF $x, $y, ([single]$size.Width), ([single]$size.Height)

        $bg = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(136, 0, 0, 0))
        $fg = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::White)
        $g.FillRectangle($bg, $rect)
        $g.DrawString($label, $font, $fg, $x, $y)
    }
    finally {
        if ($fg) { $fg.Dispose() }
        if ($bg) { $bg.Dispose() }
        if ($font) { $font.Dispose() }
        if ($g) { $g.Dispose() }
        $src.Dispose()
    }

    $tmpOut = "$Path.label.png"
    try {
        $bmp.Save($tmpOut, [System.Drawing.Imaging.ImageFormat]::Png)
    }
    finally {
        if ($bmp) { $bmp.Dispose() }
    }
    Move-Item -LiteralPath $tmpOut -Destination $Path -Force
}

function Write-Montage([object[]]$ImagePaths, [string]$OutputPath) {
    Ensure-DrawingAssembly
    Ensure-ParentDirectory $OutputPath

    $cols = 4
    $rows = 2
    $gap = 4
    $canvasW = ($script:Width * $cols) + ($gap * ($cols - 1))
    $canvasH = ($script:Height * $rows) + ($gap * ($rows - 1))

    $canvas = New-Object System.Drawing.Bitmap $canvasW, $canvasH
    $g = [System.Drawing.Graphics]::FromImage($canvas)
    try {
        $g.Clear([System.Drawing.Color]::Black)
        for ($i = 0; $i -lt $ImagePaths.Count; $i++) {
            $path = $ImagePaths[$i]
            if (!$path) {
                continue
            }
            $col = $i % $cols
            $row = [Math]::Floor($i / $cols)
            $x = $col * ($script:Width + $gap)
            $y = $row * ($script:Height + $gap)
            $img = [System.Drawing.Image]::FromFile($path)
            try {
                $g.DrawImage($img, $x, $y, $script:Width, $script:Height)
            }
            finally {
                $img.Dispose()
            }
        }
        $canvas.Save($OutputPath, [System.Drawing.Imaging.ImageFormat]::Png)
    }
    finally {
        $g.Dispose()
        $canvas.Dispose()
    }
}

$multi = $false
$script:Highlight = $null
$script:SoloHighlight = $false
$positional = New-Object System.Collections.Generic.List[string]
$i = 0
while ($i -lt $args.Count) {
    $arg = [string]$args[$i]
    switch ($arg) {
        "--multi" { $multi = $true; $i++; continue }
        "-Multi" { $multi = $true; $i++; continue }
        "--solo" { $script:SoloHighlight = $true; $i++; continue }
        "-h" { Show-Usage; exit 0 }
        "--help" { Show-Usage; exit 0 }
        "--highlight" {
            $i++
            if ($i -ge $args.Count) { throw "--highlight needs an id or comma-separated ids" }
            $script:Highlight = [string]$args[$i]
            $i++
            continue
        }
        default {
            if ($arg.StartsWith("--highlight=")) {
                $script:Highlight = $arg.Substring("--highlight=".Length)
            }
            else {
                $positional.Add($arg)
            }
            $i++
        }
    }
}

if ($positional.Count -lt 2 -or $positional.Count -gt 4) {
    Show-Usage
    exit 2
}

$html = $positional[0]
$png = $positional[1]
$script:Width = if ($positional.Count -ge 3) { [int]$positional[2] } else { 1920 }
$script:Height = if ($positional.Count -ge 4) { [int]$positional[3] } else { 1080 }

if ($script:Width -le 0 -or $script:Height -le 0) {
    throw "width and height must be positive integers"
}

$htmlItem = Get-Item -LiteralPath $html
$script:ViewerUrl = ([System.Uri]$htmlItem.FullName).AbsoluteUri
$script:Browser = Get-BrowserPath
$script:DrawingLoaded = $false
$outPath = [System.IO.Path]::GetFullPath($png)

if (!$multi) {
    Invoke-Shot "" $outPath
    Assert-NonEmptyFile $outPath "screenshot failed (empty file)"
    $size = Format-FileSize (Get-Item -LiteralPath $outPath).Length
    Write-Host "wrote $outPath ($size)"
    exit 0
}

$tmpRoot = [System.IO.Path]::GetTempPath()
$tmp = Join-Path $tmpRoot ("cpd-shot-" + [System.Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $tmp | Out-Null

try {
    $angles = @("iso", "front", "back", "left", "right", "top", "bottom")
    $shots = @{}
    foreach ($angle in $angles) {
        $shotPath = Join-Path $tmp "$angle.png"
        Invoke-Shot $angle $shotPath
        Assert-NonEmptyFile $shotPath "screenshot failed for angle '$angle'"
        Add-ImageLabel $shotPath $angle
        $shots[$angle] = $shotPath
    }

    Write-Montage @(
        $shots["iso"], $shots["front"], $shots["back"], $null,
        $shots["right"], $shots["left"], $shots["top"], $shots["bottom"]
    ) $outPath

    Assert-NonEmptyFile $outPath "multi-angle screenshot failed"
    $size = Format-FileSize (Get-Item -LiteralPath $outPath).Length
    Write-Host "wrote $outPath (multi-angle, $size)"
}
finally {
    $resolvedTmp = [System.IO.Path]::GetFullPath($tmp)
    $resolvedRoot = [System.IO.Path]::GetFullPath($tmpRoot)
    if ($resolvedTmp.StartsWith($resolvedRoot, [System.StringComparison]::OrdinalIgnoreCase) -and
        ([System.IO.Path]::GetFileName($resolvedTmp) -like "cpd-shot-*")) {
        Remove-Item -LiteralPath $resolvedTmp -Recurse -Force
    }
}
