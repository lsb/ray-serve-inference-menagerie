#!/bin/bash

set -e

NAMESPACE="ray-serve-test"
MINIKUBE_PROFILE="ray-serve-menagerie"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-600}"

echo "Deploying Gemma service to Minikube for testing..."

kubectl config use-context "$MINIKUBE_PROFILE"

echo "Creating ConfigMap from app.py..."
kubectl create configmap gemma-service-code --from-file=app.py=gemma-service/app.py -n "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

echo "Applying Gemma RayService manifest..."
kubectl apply -f k8s-manifests/gemma-rayservice.yaml

echo "Waiting for Gemma service to be ready (timeout: ${TIMEOUT_SECONDS}s)..."
kubectl wait --for=condition=ready rayservice/gemma-service -n "$NAMESPACE" --timeout="${TIMEOUT_SECONDS}s"

echo "Waiting for Ray cluster pods to be running (timeout: ${TIMEOUT_SECONDS}s)..."
kubectl wait --for=condition=ready pod -l rayCluster=gemma-service -n "$NAMESPACE" --timeout="${TIMEOUT_SECONDS}s"

echo "Gemma service deployment status:"
kubectl get rayservice gemma-service -n "$NAMESPACE"
kubectl get pods -l rayCluster=gemma-service -n "$NAMESPACE"

echo "Setting up port forwarding..."
kubectl port-forward svc/gemma-service-svc 8000:8000 -n "$NAMESPACE" &
PORT_FORWARD_PID=$!

sleep 5

echo "Gemma service deployed successfully!"
echo ""
echo "Service is available at: http://localhost:8000"
echo "Health check: curl http://localhost:8000/health"
echo ""
echo "To stop port forwarding: kill $PORT_FORWARD_PID"
echo "To run tests: ./gemma-service/tests/test_local.sh"
echo ""
echo "Useful commands:"
echo "  kubectl logs -f deployment/gemma-service-raycluster-head -n $NAMESPACE"
echo "  kubectl get rayservice -n $NAMESPACE"
echo "  kubectl describe rayservice gemma-service -n $NAMESPACE"
