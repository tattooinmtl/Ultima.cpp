# CI naming-taboo check (Windows).
# Fails if banned identifiers appear anywhere in tracked source outside allowed
# exclusions. Patterns are constructed from fragments so this script does not
# self-trigger.

$ErrorActionPreference = 'Stop'

$p1 = 'lla' + 'ma'
$p2 = 'gg' + 'ml'
$pattern = "$p1|$p2"

$includeGlobs = @(
    '*.cpp', '*.hpp', '*.h', '*.c', '*.cc', '*.cxx',
    '*.cmake', 'CMakeLists.txt',
    '*.json', '*.yml', '*.yaml',
    '*.py', '*.sh', '*.ps1', '*.bat', '*.cmd',
    '*.go'
)

$excludeDirs = @(
    '.git',
    'build',
    'out',
    '_deps',
    'third_party',
    'node_modules',
    'docs/history'
)

$excludeFiles = @(
    'ci_taboo_check.ps1',
    'ci_taboo_check.sh',
    'roadmap*.md'
)

$hits = @()

Get-ChildItem -Path . -Recurse -File -Include $includeGlobs | Where-Object {
    $rel = Resolve-Path -Relative $_.FullName
    foreach ($d in $excludeDirs) {
        if ($rel -like "*\$d\*" -or $rel -like ".\$d\*") { return $false }
    }
    foreach ($f in $excludeFiles) {
        if ($_.Name -like $f) { return $false }
    }
    return $true
} | ForEach-Object {
    $matches = Select-String -Path $_.FullName -Pattern $pattern -CaseSensitive:$false
    if ($matches) {
        foreach ($m in $matches) {
            $hits += "$($m.Path):$($m.LineNumber): $($m.Line.Trim())"
        }
    }
}

if ($hits.Count -gt 0) {
    Write-Host "TABOO GATE FAILED - banned identifier found:" -ForegroundColor Red
    $hits | ForEach-Object { Write-Host $_ }
    exit 1
}

Write-Host "Taboo gate: clean." -ForegroundColor Green
exit 0
