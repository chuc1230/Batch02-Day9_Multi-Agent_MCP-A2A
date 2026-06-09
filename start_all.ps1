echo "Starting Registry service on port 10000..."
Start-Process uv -ArgumentList "run python -m registry" -NoNewWindow
Start-Sleep -Seconds 2

echo "Starting Tax Agent on port 10102..."
Start-Process uv -ArgumentList "run python -m tax_agent" -NoNewWindow

echo "Starting Compliance Agent on port 10103..."
Start-Process uv -ArgumentList "run python -m compliance_agent" -NoNewWindow
Start-Sleep -Seconds 3

echo "Starting Law Agent on port 10101..."
Start-Process uv -ArgumentList "run python -m law_agent" -NoNewWindow
Start-Sleep -Seconds 3

echo "Starting Customer Agent on port 10100..."
Start-Process uv -ArgumentList "run python -m customer_agent" -NoNewWindow
Start-Sleep -Seconds 2

echo ""
echo "All services started!"
echo "Run test_client.py to send a query: uv run python test_client.py"