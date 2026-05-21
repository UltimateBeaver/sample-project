# Helper to read .env
$envFile = Get-Content .env
foreach ($line in $envFile) {
    if ($line -match '^([^=]+)=(.*)$') {
        $name = $matches[1].Trim()
        $value = $matches[2].Trim()
        [Environment]::SetEnvironmentVariable($name, $value, "Process")
    }
}

# Now use the environment variables
Start-Process powershell.exe -ArgumentList "-NoExit -Command llama-server -m `"$env:LLAMACPP_PATH_MODEL`" -c 32768 -ngl 99 -fa on --port 8080 -np 1"
Start-Process powershell.exe -ArgumentList "-NoExit -Command llama-server -m `"$env:LLAMACPP_PATH_EMBEDDINGS_MODEL`" -c 2048 -ngl 99 --port 8081 --embedding"