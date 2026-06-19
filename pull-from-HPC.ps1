# Helper to read .env
$envFile = Get-Content .env
foreach ($line in $envFile) {
    if ($line -match '^([^=]+)=(.*)$') {
        $name = $matches[1].Trim()
        $value = $matches[2].Trim()
        [Environment]::SetEnvironmentVariable($name, $value, "Process")
    }
}


# --- CONFIGURATION ---
$HPC_USER = "$env:HPC_USER"
$HPC_HOST = "$env:HPC_HOST"
$LOCAL_PROJECT_ROOT = "." # Relative path to local repo

# Remote paths on the HPC cluster
$REMOTE_DUMP = "~/thesis-project/sample-project/neo4j/data/neo4j.dump"
$REMOTE_LOGS = "~/thesis-project/sample-project/logs"

# Local paths mapped from project root
$LOCAL_LOGS_DIR = "$LOCAL_PROJECT_ROOT\logs"
$LOCAL_DOCKER_DATA = "$LOCAL_PROJECT_ROOT\docker\data"     # Your local container data mount
$LOCAL_TEMP_IMPORT = "$LOCAL_PROJECT_ROOT\docker\import"   # Temporary folder for loading

# Local container identifier
$DOCKER_CONTAINER_NAME = "neo4j"

Clear-Host
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "      HPC Data & Logs Sync Automation        " -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

# ---------------------------------------------------------
# STEP 1: RAW OVERWRITE LOGS
# ---------------------------------------------------------
# Not needed anymore, since I can use vscode extension to read/write remote files directly. But leaving it here for reference.
# Write-Host "`n[1/4] Overwriting local logs directory..." -ForegroundColor Yellow
# if (Test-Path $LOCAL_LOGS_DIR) {
#     Write-Host "Removing old logs directory to ensure a raw overwrite..." -ForegroundColor Gray
#     Remove-Item -Recurse -Force $LOCAL_LOGS_DIR
# }

# Download the entire directory structure down cleanly
# scp -r "${HPC_USER}@${HPC_HOST}:${REMOTE_LOGS}" $LOCAL_PROJECT_ROOT
# Write-Host "Logs directory completely overwritten." -ForegroundColor Green

# ---------------------------------------------------------
# STEP 2: DOWNLOAD NEO4J DUMP
# ---------------------------------------------------------
Write-Host "`n[2/4] Downloading neo4j.dump from cluster..." -ForegroundColor Yellow
if (!(Test-Path $LOCAL_TEMP_IMPORT)) {
    New-Item -ItemType Directory -Path $LOCAL_TEMP_IMPORT | Out-Null
}

scp "${HPC_USER}@${HPC_HOST}:${REMOTE_DUMP}" "$LOCAL_TEMP_IMPORT\neo4j.dump"
Write-Host "Dump file downloaded successfully." -ForegroundColor Green

# ---------------------------------------------------------
# STEP 3: STOP RUNNING CONTAINER
# ---------------------------------------------------------
Write-Host "`n[3/4] Stopping local container '$DOCKER_CONTAINER_NAME' to avoid file locks..." -ForegroundColor Yellow
docker stop $DOCKER_CONTAINER_NAME

# ---------------------------------------------------------
# STEP 4: IMPORT DATA & RESTORE
# ---------------------------------------------------------
Write-Host "`n[4/4] Executing admin database restoration tool..." -ForegroundColor Yellow

# Remove old database files to ensure clean import
# $OLD_DATABASE_DIR = "$LOCAL_DOCKER_DATA\databases\neo4j"
# if (Test-Path $OLD_DATABASE_DIR) {
#     Write-Host "Removing old database directory to ensure clean import..." -ForegroundColor Gray
#     Remove-Item -Recurse -Force $OLD_DATABASE_DIR
# }

# Using a one-off disposable worker container avoids index fragmentation or permission bugs
docker run --rm `
  -v "${LOCAL_DOCKER_DATA}:/data" `
  -v "${LOCAL_TEMP_IMPORT}:/import" `
  neo4j:latest `
  neo4j-admin database load neo4j --from-path=/import --overwrite-destination=true

# ---------------------------------------------------------
# STEP 5: CLEANUP AND START
# ---------------------------------------------------------
Write-Host "`nRestarting your local Neo4j environment..." -ForegroundColor Yellow
docker start $DOCKER_CONTAINER_NAME

# Clean up the large dump file locally to save storage space
if (Test-Path "$LOCAL_TEMP_IMPORT\neo4j.dump") {
    Remove-Item "$LOCAL_TEMP_IMPORT\neo4j.dump" -Force
}

Write-Host "`n=============================================" -ForegroundColor Cyan
Write-Host " SUCCESS! Local IDE Logs and Graph are Fresh! " -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Cyan