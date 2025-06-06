#!/bin/bash


set -e  # Exit on any error

SERVICE_NAME=""
USERNAME="${USER:-default}"
NAMESPACE="default"
IMAGE_TAG="latest"
NODE_SELECTOR=""
REPLICAS=1
DRY_RUN=false
VERBOSE=false

usage() {
    cat << EOF
Usage: $0 <service_name> [OPTIONS]

Deploy Ray Serve ML services to Kubernetes cluster.

ARGUMENTS:
    service_name    Name of the service to deploy (clip, gemma, grounding-dino)

OPTIONS:
    -u, --username USER     Username prefix for deployment (default: $USER)
    -n, --namespace NS      Kubernetes namespace (default: default)
    -t, --tag TAG          Docker image tag (default: latest)
    -s, --node-selector KEY=VALUE  Node selector for GPU scheduling
    -r, --replicas NUM     Number of replicas (default: 1)
    --dry-run              Show what would be deployed without applying
    -v, --verbose          Enable verbose output
    -h, --help             Show this help message

EXAMPLES:
    $0 clip --username alice --node-selector accelerator=nvidia-tesla-t4
    $0 gemma --namespace ml-services --replicas 2
    $0 grounding-dino --dry-run

EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -u|--username)
            USERNAME="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -s|--node-selector)
            NODE_SELECTOR="$2"
            shift 2
            ;;
        -r|--replicas)
            REPLICAS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "Unknown option $1"
            usage
            exit 1
            ;;
        *)
            if [[ -z "$SERVICE_NAME" ]]; then
                SERVICE_NAME="$1"
            else
                echo "Unexpected argument: $1"
                usage
                exit 1
            fi
            shift
            ;;
    esac
done

if [[ -z "$SERVICE_NAME" ]]; then
    echo "Error: service_name is required"
    usage
    exit 1
fi

case "$SERVICE_NAME" in
    clip|gemma|grounding-dino)
        ;;
    *)
        echo "Error: Invalid service name '$SERVICE_NAME'. Must be one of: clip, gemma, grounding-dino"
        exit 1
        ;;
esac

log() {
    if [[ "$VERBOSE" == "true" ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    fi
}

if ! command -v kubectl &> /dev/null; then
    echo "Error: kubectl is not installed or not in PATH"
    exit 1
fi

if ! kubectl cluster-info &> /dev/null; then
    echo "Error: Cannot connect to Kubernetes cluster. Check your kubeconfig."
    exit 1
fi

log "Deploying $SERVICE_NAME service for user $USERNAME"
log "Namespace: $NAMESPACE"
log "Image tag: $IMAGE_TAG"
log "Replicas: $REPLICAS"

if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
    log "Creating namespace $NAMESPACE"
    if [[ "$DRY_RUN" == "false" ]]; then
        kubectl create namespace "$NAMESPACE"
    else
        echo "Would create namespace: $NAMESPACE"
    fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST_DIR="$SCRIPT_DIR/../k8s-manifests"

DEPLOYMENT_FILE="$MANIFEST_DIR/${SERVICE_NAME}-deployment.yaml"
SERVICE_FILE="$MANIFEST_DIR/${SERVICE_NAME}-service.yaml"

if [[ ! -f "$DEPLOYMENT_FILE" ]]; then
    echo "Error: Deployment manifest not found: $DEPLOYMENT_FILE"
    exit 1
fi

if [[ ! -f "$SERVICE_FILE" ]]; then
    echo "Error: Service manifest not found: $SERVICE_FILE"
    exit 1
fi

TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

TEMP_DEPLOYMENT="$TEMP_DIR/deployment.yaml"
TEMP_SERVICE="$TEMP_DIR/service.yaml"

sed -e "s/\${USERNAME}/$USERNAME/g" \
    -e "s/\${SERVICE_NAME}/$SERVICE_NAME/g" \
    -e "s/YOUR_REGISTRY\/.*:latest/YOUR_REGISTRY\/${SERVICE_NAME}-service:${IMAGE_TAG}/g" \
    -e "s/replicas: 1/replicas: $REPLICAS/g" \
    "$DEPLOYMENT_FILE" > "$TEMP_DEPLOYMENT"

if [[ -n "$NODE_SELECTOR" ]]; then
    KEY=$(echo "$NODE_SELECTOR" | cut -d'=' -f1)
    VALUE=$(echo "$NODE_SELECTOR" | cut -d'=' -f2)
    
    sed -i "/nodeSelector:/,/accelerator:/ {
        /accelerator:/c\\
        $KEY: \"$VALUE\"
    }" "$TEMP_DEPLOYMENT"
fi

sed -e "s/\${USERNAME}/$USERNAME/g" \
    -e "s/\${SERVICE_NAME}/$SERVICE_NAME/g" \
    "$SERVICE_FILE" > "$TEMP_SERVICE"

if [[ "$DRY_RUN" == "true" || "$VERBOSE" == "true" ]]; then
    echo "=== Deployment manifest ==="
    cat "$TEMP_DEPLOYMENT"
    echo ""
    echo "=== Service manifest ==="
    cat "$TEMP_SERVICE"
    echo ""
fi

if [[ "$DRY_RUN" == "false" ]]; then
    log "Applying deployment manifest"
    kubectl apply -f "$TEMP_DEPLOYMENT" -n "$NAMESPACE"
    
    log "Applying service manifest"
    kubectl apply -f "$TEMP_SERVICE" -n "$NAMESPACE"
    
    echo "Successfully deployed $SERVICE_NAME service!"
    echo ""
    echo "Check deployment status with:"
    echo "  kubectl get pods -n $NAMESPACE -l app=$USERNAME-$SERVICE_NAME"
    echo ""
    echo "View logs with:"
    echo "  kubectl logs -n $NAMESPACE -l app=$USERNAME-$SERVICE_NAME -f"
    echo ""
    echo "Port forward to test locally:"
    echo "  kubectl port-forward -n $NAMESPACE svc/$USERNAME-$SERVICE_NAME 8080:80"
else
    echo "Dry run completed. Use --dry-run=false to actually deploy."
fi
