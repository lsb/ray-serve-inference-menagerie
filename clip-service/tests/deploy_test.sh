#!/bin/bash

set -e

NAMESPACE="ray-serve-test"
MINIKUBE_PROFILE="ray-serve-menagerie"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-600}"  # Default to 10 minutes, can be overridden

echo "Deploying CLIP service to Minikube for testing..."

kubectl config use-context "$MINIKUBE_PROFILE"

echo "Creating ConfigMap from app.py..."
kubectl create configmap clip-service-code --from-file=app.py=clip-service/app.py -n "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

echo "Applying CLIP RayService manifest..."
kubectl apply -f k8s-manifests/clip-rayservice.yaml

echo "Waiting for CLIP service to be ready (timeout: ${TIMEOUT_SECONDS}s)..."
kubectl wait --for=condition=ready rayservice/clip-service -n "$NAMESPACE" --timeout="${TIMEOUT_SECONDS}s"

echo "Waiting for Ray cluster pods to be running (timeout: ${TIMEOUT_SECONDS}s)..."
kubectl wait --for=condition=ready pod -l rayCluster=clip-service -n "$NAMESPACE" --timeout="${TIMEOUT_SECONDS}s"

echo "CLIP service deployment status:"
kubectl get rayservice clip-service -n "$NAMESPACE"
kubectl get pods -l rayCluster=clip-service -n "$NAMESPACE"

echo "Setting up port forwarding..."
kubectl port-forward svc/clip-service-svc 8000:8000 -n "$NAMESPACE" &
PORT_FORWARD_PID=$!

sleep 5

echo "CLIP service deployed successfully!"
echo ""
echo "Service is available at: http://localhost:8000"
echo "Health check: curl http://localhost:8000/health"
echo ""
echo "To stop port forwarding: kill $PORT_FORWARD_PID"
echo "To run tests: ./clip-service/tests/test_local.sh"
echo ""
echo "Useful commands:"
echo "  kubectl logs -f deployment/clip-service-raycluster-head -n $NAMESPACE"
echo "  kubectl get rayservice -n $NAMESPACE"
echo "  kubectl describe rayservice clip-service -n $NAMESPACE"
