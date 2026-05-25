# Push this folder to GitHub
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "=== Folder contents ==="
Get-ChildItem -Force | Format-Table Name, Length

Write-Host "`n=== GitHub auth ==="
gh auth status
$username = gh api user -q .login
Write-Host "GitHub user: $username"

if (-not (Test-Path .git)) {
    Write-Host "`n=== Initializing git repo ==="
    git init
}

Write-Host "`n=== Staging and committing ==="
git add -A
git status
$status = git status --porcelain
if ($status) {
    git commit -m "Initial commit: bank simulation central policy models"
} else {
    Write-Host "Nothing to commit (already committed?)"
}

Write-Host "`n=== Creating/pushing GitHub repo ==="
$repoName = "central-bank-simulation"
$remoteUrl = git remote get-url origin 2>$null
if (-not $remoteUrl) {
    gh repo create $repoName --public `
        --description "Bank simulation: central policy, decentralized danger zone, interbank rollover" `
        --source=. --remote=origin --push
} else {
    Write-Host "Remote already exists: $remoteUrl"
    git push -u origin HEAD
}

Write-Host "`n=== Done ==="
git remote get-url origin
Write-Host "URL: https://github.com/$username/$repoName"
