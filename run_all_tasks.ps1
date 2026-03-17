param(
    [switch]$LaunchApps,
    [switch]$RunTask5Training,
    [switch]$StopOnError
)

$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $RootDir ".venv\Scripts\python.exe"
$VenvStreamlit = Join-Path $RootDir ".venv\Scripts\streamlit.exe"

$PythonExe = if (Test-Path $VenvPython) { $VenvPython } else { "python" }
$StreamlitExe = if (Test-Path $VenvStreamlit) { $VenvStreamlit } else { "streamlit" }

$results = @()

function Invoke-Step {
    param(
        [string]$Name,
        [string]$WorkingDir,
        [string]$Script,
        [string[]]$CommandArgs
    )

    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    Push-Location $WorkingDir
    try {
        & $Script @CommandArgs
        if ($LASTEXITCODE -eq 0) {
            Write-Host "PASS: $Name" -ForegroundColor Green
            $script:results += [pscustomobject]@{ Task = $Name; Status = "PASS" }
        }
        else {
            Write-Host "FAIL: $Name (exit code $LASTEXITCODE)" -ForegroundColor Red
            $script:results += [pscustomobject]@{ Task = $Name; Status = "FAIL" }
            if ($StopOnError) {
                throw "Stopping due to -StopOnError"
            }
        }
    }
    catch {
        Write-Host "FAIL: $Name ($($_.Exception.Message))" -ForegroundColor Red
        $script:results += [pscustomobject]@{ Task = $Name; Status = "FAIL" }
        if ($StopOnError) {
            throw
        }
    }
    finally {
        Pop-Location
    }
}

Write-Host "Using Python: $PythonExe" -ForegroundColor Yellow
Write-Host "Root folder:  $RootDir" -ForegroundColor Yellow

Invoke-Step -Name "Task 1 Verification" -WorkingDir (Join-Path $RootDir "task1_iris_exploration") -Script $PythonExe -CommandArgs @("task1_solution.py")
Invoke-Step -Name "Task 2 Verification" -WorkingDir (Join-Path $RootDir "task2_stock_prediction") -Script $PythonExe -CommandArgs @("task2_solution.py")
Invoke-Step -Name "Task 3 Verification" -WorkingDir (Join-Path $RootDir "task3_heart_disease_prediction") -Script $PythonExe -CommandArgs @("task3_solution.py")
Invoke-Step -Name "Task 4 Verification" -WorkingDir (Join-Path $RootDir "task4_health_query_chatbot") -Script $PythonExe -CommandArgs @("-c", "from chatbot import HealthChatbot; b=HealthChatbot(); p=b.get_provider(); print('provider=', p); assert p != 'none', 'No API provider configured from environment variables'; print(b.ask('What causes a sore throat?')[:200])")

if ($RunTask5Training) {
    Invoke-Step -Name "Task 5 Training" -WorkingDir (Join-Path $RootDir "task5_mental_health_chatbot_finetuned") -Script $PythonExe -CommandArgs @("train.py")
}
else {
    Write-Host "`nSkipping Task 5 training (use -RunTask5Training to enable)." -ForegroundColor DarkYellow
}

Invoke-Step -Name "Task 5 Inference Verification" -WorkingDir (Join-Path $RootDir "task5_mental_health_chatbot_finetuned") -Script $PythonExe -CommandArgs @("-c", "from chatbot import MentalHealthSupportBot; b=MentalHealthSupportBot(); print(b.generate('I feel stressed about exams.')[:200])")
Invoke-Step -Name "Task 6 Verification" -WorkingDir (Join-Path $RootDir "task6_house_price_prediction") -Script $PythonExe -CommandArgs @("task6_solution.py")

Write-Host "`n=== Summary ===" -ForegroundColor Cyan
$results | Format-Table -AutoSize

if ($LaunchApps) {
    Write-Host "`nLaunching all Streamlit apps in separate windows..." -ForegroundColor Yellow

    $apps = @(
        @{ Dir = "task1_iris_exploration"; Port = 8501 },
        @{ Dir = "task2_stock_prediction"; Port = 8502 },
        @{ Dir = "task3_heart_disease_prediction"; Port = 8503 },
        @{ Dir = "task4_health_query_chatbot"; Port = 8504 },
        @{ Dir = "task5_mental_health_chatbot_finetuned"; Port = 8505 },
        @{ Dir = "task6_house_price_prediction"; Port = 8506 }
    )

    foreach ($app in $apps) {
        $dirPath = Join-Path $RootDir $app.Dir
        Start-Process powershell -ArgumentList @(
            "-NoExit",
            "-Command",
            "Set-Location '$dirPath'; & '$StreamlitExe' run app.py --server.headless true --server.port $($app.Port)"
        )
        Write-Host "Started $($app.Dir) on http://localhost:$($app.Port)" -ForegroundColor Green
    }
}

$failed = @($results | Where-Object { $_.Status -eq "FAIL" })
if ($failed.Count -gt 0) {
    exit 1
}

Write-Host "`nAll verification steps completed successfully." -ForegroundColor Green
exit 0
