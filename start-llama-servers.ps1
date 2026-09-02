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
# Commented old command with reasoning implicitly enabled
#Start-Process powershell.exe -ArgumentList "-NoExit -Command llama-server -m `"$env:LLAMACPP_PATH_MODEL`" -c `"$env:LLAMACPP_MODEL_CONTEXT_SIZE`" -ngl `"$env:LLAMACPP_MODEL_NGL`" -fa on --port `"$env:LLAMA_CPP_MODEL_PORT`" -np `"$env:LLAMACPP_MODEL_NUM_PARALLEL_SLOTS`" --no-mmap --reasoning off --chat-template-kwargs `"`"{\`"enable_thinking\`":false}\`"`""

# Reasoning setting globally
#$scriptBlock = @"
#llama-server -m "$env:LLAMACPP_PATH_MODEL" -c "$env:LLAMACPP_MODEL_CONTEXT_SIZE" -ngl "$env:LLAMACPP_MODEL_NGL" -fa on --port "$env:LLAMA_CPP_MODEL_PORT" -np "$env:LLAMACPP_MODEL_NUM_PARALLEL_SLOTS" --no-mmap --reasoning "$env:LLAMACPP_MODEL_REASONING" --chat-template-kwargs '{\"enable_thinking\":"$env:LLAMACPP_MODEL_THINKING"}'
#"@

# Reasoning setting left as default (true)
$scriptBlock = @"
llama-server -m "$env:LLAMACPP_PATH_MODEL" -c "$env:LLAMACPP_MODEL_CONTEXT_SIZE" -ngl "$env:LLAMACPP_MODEL_NGL" -fa on --port "$env:LLAMA_CPP_MODEL_PORT" -np "$env:LLAMACPP_MODEL_NUM_PARALLEL_SLOTS" --load-mode mmap
"@

# Convert to Base64 (bypasses all character/quote parsing)
$bytes = [System.Text.Encoding]::Unicode.GetBytes($scriptBlock)
$encodedCommand = [Convert]::ToBase64String($bytes)

# Launch llm server in a new window
Start-Process powershell.exe -ArgumentList "-NoExit", "-EncodedCommand", $encodedCommand
# Launch embedding server in a new window
Start-Process powershell.exe -ArgumentList "-NoExit -Command llama-server -m `"$env:LLAMACPP_PATH_EMBEDDINGS_MODEL`" -c `"$env:LLAMACPP_EMBED_CONTEXT_SIZE`" -ngl `"$env:LLAMACPP_EMBED_NGL`" --port `"$env:LLAMA_CPP_EMBED_PORT`" --embedding --load-mode mmap --pooling `"$env:LLAMACPP_EMBED_POOLING`" "