#!/bin/bash
set -e

echo "=========================================="
echo "LLM Serving Framework Setup"
echo "=========================================="

# Check CUDA availability
echo "Checking CUDA installation..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Please install CUDA drivers."
    exit 1
fi

echo "CUDA Status:"
nvidia-smi

# Check Docker
echo -e "\nChecking Docker installation..."
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker not found. Please install Docker."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "ERROR: Docker Compose not found. Please install Docker Compose."
    exit 1
fi

echo "Docker version:"
docker --version
docker-compose --version

# Create necessary directories
echo -e "\nCreating directories..."
mkdir -p models logs config/grafana/dashboards config/grafana/datasources

# Copy environment file
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "Please edit .env file with your configuration."
else
    echo ".env file already exists."
fi

# Create config files
echo -e "\nCreating configuration files..."

# Grafana datasource
cat > config/grafana/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: false
EOF

# NGINX config
cat > config/nginx.conf << EOF
events {
    worker_connections 1024;
}

http {
    upstream llm_servers {
        least_conn;
        server llm-server:8000;
    }

    server {
        listen 80;
        
        location /health {
            proxy_pass http://llm_servers/health;
        }
        
        location / {
            proxy_pass http://llm_servers;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_read_timeout 300s;
            proxy_connect_timeout 75s;
        }
    }
}
EOF

# Prometheus alerts
cat > config/alerts.yml << EOF
groups:
  - name: llm_alerts
    interval: 30s
    rules:
      - alert: HighLatency
        expr: llm_inference_latency_seconds{quantile="0.99"} > 0.5
        for: 5m
        annotations:
          summary: "High inference latency detected"
          
      - alert: HighErrorRate
        expr: rate(llm_errors_total[5m]) > 0.1
        for: 5m
        annotations:
          summary: "High error rate detected"
          
      - alert: GPUMemoryHigh
        expr: llm_gpu_memory_usage_bytes > 20e9
        for: 5m
        annotations:
          summary: "GPU memory usage is high"
EOF

echo -e "\n=========================================="
echo "Setup complete!"
echo "=========================================="
echo -e "\nNext steps:"
echo "1. Edit .env file with your model and configuration"
echo "2. Run: docker-compose build"
echo "3. Run: docker-compose up -d"
echo "4. Check health: curl http://localhost:8000/health"
echo "5. Access Grafana: http://localhost:3000 (admin/admin)"
echo "=========================================="
