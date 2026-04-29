<#
.SYNOPSIS
    yawnbot dev 환경 자동 세팅: cloudflared + gh CLI 설치 + gh 로그인.

.DESCRIPTION
    GitHub webhook 자동 등록(scripts/tunnel-launcher.mjs) 에 필요한 외부 도구를
    한 번에 갖춰 주는 멱등 스크립트. 이미 설치/로그인된 항목은 건너뛴다.

    실행: npm run setup:env
        또는 직접: powershell -ExecutionPolicy Bypass -File scripts/setup-env.ps1
#>

$ErrorActionPreference = 'Stop'

function Test-Cmd([string]$name) {
    return [bool](Get-Command $name -ErrorAction SilentlyContinue)
}

function Install-IfMissing([string]$cmd, [string]$wingetId, [string]$label) {
    if (Test-Cmd $cmd) {
        Write-Host "[setup] $label 이미 설치됨 - skip"
        return $false
    }
    Write-Host "[setup] $label 설치 중 (winget id=$wingetId)..."
    winget install --id $wingetId --silent --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        throw "[setup] $label 설치 실패 (winget exit $LASTEXITCODE)"
    }
    return $true
}

# winget 사전 확인
if (-not (Test-Cmd 'winget')) {
    Write-Error "[setup] winget 이 없습니다. Microsoft Store > 'App Installer' 를 먼저 설치해주세요."
    exit 1
}

# 1) 패키지 설치
$installedAny = $false
if (Install-IfMissing 'cloudflared' 'Cloudflare.cloudflared' 'cloudflared') { $installedAny = $true }
if (Install-IfMissing 'gh' 'GitHub.cli' 'GitHub CLI') { $installedAny = $true }

# winget 으로 새로 깔린 도구는 같은 세션 PATH 에 즉시 안 잡힘 - refresh
if ($installedAny) {
    $machinePath = [System.Environment]::GetEnvironmentVariable('Path', 'Machine')
    $userPath = [System.Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = "$machinePath;$userPath"
}

if (-not (Test-Cmd 'cloudflared')) {
    Write-Error "[setup] cloudflared 가 PATH 에 잡히지 않습니다. 새 PowerShell 창을 열고 다시 실행해주세요."
    exit 1
}
if (-not (Test-Cmd 'gh')) {
    Write-Error "[setup] gh 가 PATH 에 잡히지 않습니다. 새 PowerShell 창을 열고 다시 실행해주세요."
    exit 1
}

# 2) gh 로그인 상태 확인 -> 미로그인 시 진행
& gh auth status 2>$null | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "[setup] gh 이미 로그인됨 - skip"
}
else {
    Write-Host "[setup] gh 로그인 시작 (브라우저로 인증)..."
    & gh auth login --hostname github.com --git-protocol https --web
    if ($LASTEXITCODE -ne 0) {
        Write-Error "[setup] gh 로그인 실패"
        exit 1
    }
}

# 3) 최종 상태 출력
Write-Host ""
Write-Host "[setup] === 설치 결과 ==="
& cloudflared --version
& gh --version | Select-Object -First 1
& gh auth status

Write-Host ""
Write-Host "[setup] 완료. 다음 단계:"
Write-Host "  - data/webhook-routes.json 에 githubRepos 추가"
Write-Host "  - npm run tunnel 로 cloudflared + webhook 자동 등록 실행"
