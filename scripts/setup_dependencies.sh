#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Setting up Ray Serve Inference Menagerie dependencies..."

install_system_deps() {
    echo "Installing system dependencies..."
    
    sudo apt-get update
    
    if ! command -v docker &> /dev/null; then
        echo "Installing Docker..."
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        sudo usermod -aG docker $USER
        rm get-docker.sh
        echo "Docker installed. You may need to log out and back in for group changes to take effect."
    else
        echo "Docker already installed"
    fi
    
    if ! command -v kubectl &> /dev/null; then
        echo "Installing kubectl..."
        curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
        sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
        rm kubectl
    else
        echo "kubectl already installed"
    fi
    
    if ! command -v minikube &> /dev/null; then
        echo "Installing minikube..."
        curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
        sudo install minikube-linux-amd64 /usr/local/bin/minikube
        rm minikube-linux-amd64
    else
        echo "minikube already installed"
    fi
    
    if ! command -v helm &> /dev/null; then
        echo "Installing Helm..."
        curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
    else
        echo "Helm already installed"
    fi
    
    echo "System dependencies installed successfully"
}

setup_python_env() {
    echo "Setting up Python environment..."
    
    cd "$PROJECT_ROOT"
    
    if [ ! -d "venv" ]; then
        echo "Creating Python virtual environment..."
        python3 -m venv venv
    fi
    
    source venv/bin/activate
    
    pip install --upgrade pip
    
    echo "Installing Python dependencies..."
    pip install \
        "ray[serve]>=2.8.0" \
        "torch>=2.0.0" \
        "transformers>=4.30.0" \
        "Pillow>=9.0.0" \
        "requests>=2.28.0" \
        "fastapi>=0.100.0" \
        "uvicorn>=0.20.0" \
        "python-multipart>=0.0.6"
    
    if [[ "${1:-}" == "--dev" ]]; then
        echo "Installing development dependencies..."
        pip install \
            "pytest>=7.0.0" \
            "pytest-asyncio>=0.21.0" \
            "black>=23.0.0" \
            "flake8>=6.0.0" \
            "mypy>=1.0.0" \
            "pre-commit>=3.0.0"
    fi
    
    echo "Python environment setup completed"
}

install_system_deps
setup_python_env "$@"

echo "All dependencies installed successfully!"
echo ""
echo "Next steps:"
echo "1. Activate the Python environment: source venv/bin/activate"
echo "2. Set up Minikube and KubeRay: ./scripts/setup_minikube.sh"
echo "3. Run tests: ./scripts/test_local.sh"
