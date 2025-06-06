#!/bin/bash

set -e

echo "Setting up Minikube with KubeRay for Ray Serve Inference Menagerie..."

MINIKUBE_PROFILE="ray-serve-menagerie"
MINIKUBE_MEMORY="8192"
MINIKUBE_CPUS="4"
KUBERAY_VERSION="1.0.0"

setup_minikube() {
    echo "Setting up Minikube cluster..."
    
    if minikube profile list | grep -q "$MINIKUBE_PROFILE"; then
        echo "Minikube profile '$MINIKUBE_PROFILE' already exists. Deleting and recreating..."
        minikube delete -p "$MINIKUBE_PROFILE"
    fi
    
    echo "Starting Minikube with profile: $MINIKUBE_PROFILE"
    minikube start \
        --profile="$MINIKUBE_PROFILE" \
        --memory="$MINIKUBE_MEMORY" \
        --cpus="$MINIKUBE_CPUS" \
        --driver=docker \
        --kubernetes-version=v1.28.3
    
    kubectl config use-context "$MINIKUBE_PROFILE"
    
    echo "Minikube cluster started successfully"
}

install_kuberay() {
    echo "Installing KubeRay operator..."
    
    helm repo add kuberay https://ray-project.github.io/kuberay-helm/
    helm repo update
    
    kubectl create namespace kuberay-operator --dry-run=client -o yaml | kubectl apply -f -
    
    helm upgrade --install kuberay-operator kuberay/kuberay-operator \
        --namespace kuberay-operator \
        --version "$KUBERAY_VERSION" \
        --wait
    
    echo "Waiting for KubeRay operator to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/kuberay-operator -n kuberay-operator
    
    echo "KubeRay operator installed successfully"
}

verify_setup() {
    echo "Verifying setup..."
    
    minikube status -p "$MINIKUBE_PROFILE"
    
    kubectl get pods -n kuberay-operator
    
    kubectl top nodes || echo "Metrics server not available (this is normal for basic setup)"
    
    echo "Setup verification completed"
}

create_test_namespace() {
    echo "Creating test namespace..."
    kubectl create namespace ray-serve-test --dry-run=client -o yaml | kubectl apply -f -
    echo "Test namespace created"
}

setup_minikube
install_kuberay
create_test_namespace
verify_setup

echo ""
echo "Minikube and KubeRay setup completed successfully!"
echo ""
echo "Cluster info:"
echo "  Profile: $MINIKUBE_PROFILE"
echo "  Memory: ${MINIKUBE_MEMORY}MB"
echo "  CPUs: $MINIKUBE_CPUS"
echo "  KubeRay version: $KUBERAY_VERSION"
echo ""
echo "Next steps:"
echo "1. Deploy CLIP service: ./clip-service/tests/deploy_test.sh"
echo "2. Run tests: ./clip-service/tests/test_local.sh"
echo ""
echo "Useful commands:"
echo "  minikube dashboard -p $MINIKUBE_PROFILE  # Open Kubernetes dashboard"
echo "  kubectl get rayservices -A              # List Ray services"
echo "  kubectl logs -f <pod-name>              # View pod logs"
