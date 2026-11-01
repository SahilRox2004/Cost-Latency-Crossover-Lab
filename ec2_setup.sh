#!/bin/bash
# ========================================================
# EC2 Setup Script for ML Inference Cost-Latency Lab
# Target OS: Ubuntu 22.04 LTS / Ubuntu 24.04 LTS
# ========================================================

set -e

# Update and install dependencies
sudo apt-get update -y
sudo apt-get install -y python3-pip python3-venv git

# Create app directory if running directly
mkdir -p /home/ubuntu/app
cd /home/ubuntu/app

# If your project is in a GitHub repository:
# git clone <YOUR_REPO_URL> .

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip & install requirements
pip install --upgrade pip
pip install -r requirements.txt

# Create systemd service for automatic reboot recovery & persistent background running
sudo bash -c 'cat <<EOF > /etc/systemd/system/streamlit.service
[Unit]
Description=Streamlit App
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/app
ExecStart=/home/ubuntu/app/venv/bin/streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF'

# Reload systemd, enable and start service
sudo systemctl daemon-reload
sudo systemctl enable streamlit.service
sudo systemctl start streamlit.service

echo "Streamlit service started! Check status with: sudo systemctl status streamlit.service"
