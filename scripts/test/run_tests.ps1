# PowerShell script for running tests with proper conditional logic
param(
    [string]$TestType = "unit",
    [string]$DockerTests = $env:DOCKER_TESTS,
    [string]$PerformanceTests = $env:PERFORMANCE_TESTS
)

Write-Host "Running $TestType tests..."

switch ($TestType) {
    "containerized" {
        if ($DockerTests -eq "true") {
            Write-Host "Running containerized tests..."
            uv run pytest tests/ -m containerized -v --tb=short
        } else {
            Write-Host "Containerized tests skipped (set DOCKER_TESTS=true to enable)"
        }
    }
    "docker" {
        if ($DockerTests -eq "true") {
            Write-Host "Running Docker sandbox tests..."
            uv run pytest tests/test_docker_sandbox/ -v --tb=short
        } else {
            Write-Host "Docker tests skipped (set DOCKER_TESTS=true to enable)"
        }
    }
    "bioinformatics" {
        if ($DockerTests -eq "true") {
            Write-Host "Running bioinformatics tools tests..."
            uv run pytest tests/test_bioinformatics_tools/ -v --tb=short
        } else {
            Write-Host "Bioinformatics tests skipped (set DOCKER_TESTS=true to enable)"
        }
    }
    "unit" {
        Write-Host "Running unit tests..."
        uv run pytest tests/ -m "unit" -v
    }
    "integration" {
        Write-Host "Running integration tests..."
        uv run pytest tests/ -m "integration" -v
    }
    "performance" {
        if ($PerformanceTests -eq "true") {
            Write-Host "Running performance tests with benchmarks..."
            uv run pytest tests/ -m performance --benchmark-only --benchmark-json=benchmark.json
        } else {
            Write-Host "Running performance tests..."
            uv run pytest tests/test_performance/ -v
        }
    }
    default {
        Write-Host "Running $TestType tests..."
        uv run pytest tests/ -m $TestType -v
    }
}
