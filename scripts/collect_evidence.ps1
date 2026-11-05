# Collects implementation evidence for Chapter 5 into reports\evidence\.
# Run from an activated backend virtual environment:
#   cd C:\mzazicare-backend
#   powershell -ExecutionPolicy Bypass -File scripts\collect_evidence.ps1
# It never reads .env files and strips credentials from Git remote URLs.

$ErrorActionPreference = 'Continue'
$Backend  = 'C:\mzazicare-backend'
$Frontend = 'C:\mzazicare'
$Out = Join-Path $Backend 'reports\evidence'
New-Item -ItemType Directory -Force -Path $Out | Out-Null

function Save($name, [scriptblock]$block) {
    Write-Host "Collecting $name ..."
    try { & $block *>&1 | Out-File -FilePath (Join-Path $Out $name) -Encoding utf8 }
    catch { "ERROR: $_" | Out-File -FilePath (Join-Path $Out $name) -Encoding utf8 }
}

function Clean-Remote($text) { $text -replace '://[^/@\s]+@', '://' }

# 1. Automated tests
Save 'backend_tests.txt' {
    Push-Location $Backend
    python -X utf8 -m unittest discover -s tests -p "test_*.py" -v
    "EXIT CODE: $LASTEXITCODE"
    Pop-Location
}
Save 'frontend_tests.txt' {
    Push-Location $Frontend
    $files = (Get-ChildItem -Filter '*.test.cjs').Name
    node --test @files
    "EXIT CODE: $LASTEXITCODE"
    Pop-Location
}

# 2. Version control history
foreach ($repo in @(@{Name='backend'; Path=$Backend}, @{Name='frontend'; Path=$Frontend})) {
    Save "git_$($repo.Name).txt" {
        Push-Location $repo.Path
        '== remotes =='
        git remote -v | ForEach-Object { Clean-Remote $_ }
        '== branches =='
        git branch -a
        '== tags =='
        git tag --list
        '== status (summary) =='
        git status --short --branch
        '== commit count =='
        git rev-list --count HEAD
        '== log (hash|date|subject) =='
        git log --date=short --pretty=format:'%h|%ad|%s'
        Pop-Location
    }
}

# 3. Environment and versions
Save 'environment.txt' {
    '== host =='
    $os  = Get-CimInstance Win32_OperatingSystem
    $cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
    $cs  = Get-CimInstance Win32_ComputerSystem
    "OS: $($os.Caption) $($os.Version)"
    "CPU: $($cpu.Name) ($($cpu.NumberOfCores) cores, $($cpu.NumberOfLogicalProcessors) threads)"
    "RAM: {0:N1} GB" -f ($cs.TotalPhysicalMemory / 1GB)
    '== runtimes =='
    "python: $(python --version 2>&1)"
    "node: $(node --version 2>&1)"
    "npm: $(npm --version 2>&1)"
    "docker: $(docker version --format '{{.Server.Version}}' 2>&1)"
    "docker compose: $(docker compose version --short 2>&1)"
    "cloudflared: $(cloudflared --version 2>&1)"
    "postgres (container): $(docker exec mzazicare-postgres postgres --version 2>&1)"
    '== containers =='
    docker ps --format '{{.Names}} | {{.Image}} | {{.Status}} | {{.Ports}}'
    '== key python packages =='
    python -m pip list 2>$null | Select-String -Pattern '^(fastapi|uvicorn|sqlalchemy|asyncpg|alembic|scikit-learn|shap|pandas|numpy|joblib|cryptography|slowapi|structlog|python-jose|PyJWT|google-auth) '
    '== key frontend packages (package.json) =='
    $pkg = Get-Content (Join-Path $Frontend 'package.json') -Raw | ConvertFrom-Json
    foreach ($n in 'expo','react-native','react','expo-local-authentication','expo-secure-store','expo-screen-capture','@react-native-google-signin/google-signin','@react-navigation/native') {
        "${n}: $($pkg.dependencies.$n)"
    }
    '== Android device (adb) =='
    adb devices -l
    "model: $(adb shell getprop ro.product.model 2>&1)"
    "android: $(adb shell getprop ro.build.version.release 2>&1) (API $(adb shell getprop ro.build.version.sdk 2>&1))"
}

# 4. Health check through the Cloudflare tunnel (20 requests)
Save 'health_check.txt' {
    $url = 'https://api.mzazicare.bond/api/v1/health'
    $times = @()
    for ($i = 1; $i -le 20; $i++) {
        $sw = [Diagnostics.Stopwatch]::StartNew()
        try { $r = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 15; $code = $r.StatusCode; $body = $r.Content }
        catch { $code = 'ERROR'; $body = $_.Exception.Message }
        $sw.Stop(); $times += $sw.Elapsed.TotalMilliseconds
        "request $i : $code  $([math]::Round($sw.Elapsed.TotalMilliseconds,1)) ms"
    }
    "last body: $body"
    $sorted = $times | Sort-Object
    "median ms: $([math]::Round($sorted[[int]($sorted.Count/2)],1))"
    "p95 ms: $([math]::Round($sorted[[int][math]::Ceiling($sorted.Count*0.95)-1],1))"
}

Write-Host "Done. Files written to $Out"
