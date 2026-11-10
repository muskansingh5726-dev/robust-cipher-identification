param(
    [string]$MetadataPath = "data/source_metadata.csv",
    [string]$OutputDirectory = "data/sources"
)

$ErrorActionPreference = "Stop"
$metadata = Import-Csv -LiteralPath $MetadataPath
if (-not $metadata) {
    throw "The metadata file contains no source rows: $MetadataPath"
}

$outputRoot = [System.IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$downloaded = 0

foreach ($source in $metadata) {
    if ([string]::IsNullOrWhiteSpace($source.filename) -or
        [string]::IsNullOrWhiteSpace($source.download_url)) {
        throw "Every metadata row needs filename and download_url."
    }
    if ($source.filename -match '[/\\]' -or $source.filename -match '\.\.') {
        throw "Expected a simple filename, got: $($source.filename)"
    }
    if ($source.download_url -notmatch '^https://gutenberg\.pglaf\.org/cache/epub/\d+/pg\d+\.txt$') {
        throw "Download URL is not an approved Project Gutenberg mirror URL: $($source.download_url)"
    }

    $target = [System.IO.Path]::GetFullPath(
        [System.IO.Path]::Combine($outputRoot, $source.filename)
    )
    if (-not $target.StartsWith(
        $outputRoot + [System.IO.Path]::DirectorySeparatorChar,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Unsafe output filename: $($source.filename)"
    }

    $request = @{
        Uri       = $source.download_url
        OutFile   = $target
        UserAgent = "RobustCipherResearchDataset/1.0"
        TimeoutSec = 120
    }
    Invoke-WebRequest @request
    if ((Get-Item -LiteralPath $target).Length -lt 1000) {
        throw "Downloaded file is unexpectedly small: $($source.filename)"
    }
    $downloaded++
    Start-Sleep -Milliseconds 500
}

Write-Host "Downloaded $downloaded source documents to $outputRoot"
