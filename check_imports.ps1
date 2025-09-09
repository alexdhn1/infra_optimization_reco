# check_imports.ps1
# Script pour scanner tous les fichiers .py et détecter les imports relatifs

$root = Get-Location
Write-Host "🔍 Scan des imports relatifs dans $root`n"

Get-ChildItem -Recurse -Include *.py | ForEach-Object {
    $file = $_.FullName
    $lines = Get-Content $file
    $matches = $lines | Select-String -Pattern "from\s+\.\.?" -SimpleMatch
    if ($matches) {
        Write-Host "⚠️  Fichier: $file"
        foreach ($m in $matches) {
            Write-Host "    Ligne $($m.LineNumber): $($m.Line.Trim())"
        }
        Write-Host ""
    }
}

Write-Host "`n✅ Scan terminé."
