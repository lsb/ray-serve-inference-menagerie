#!/bin/bash

set -e

NAMESPACE="ray-serve-test"
MINIKUBE_PROFILE="ray-serve-menagerie"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-600}"

echo "Deploying Grounding DINO service to Minikube for testing..."

kubectl config use-context "$MINIKUBE_PROFILE"

echo "Creating ConfigMap from app.py..."
kubectl create configmap grounding-dino-service-code --from-file=app.py=grounding-dino-service/app.py -n "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

echo "Applying Grounding DINO RayService manifest..."
kubectl apply -f k8s-manifests/grounding-dino-rayservice.yaml

echo "Waiting for Grounding DINO service to be ready (timeout: ${TIMEOUT_SECONDS}s)..."
kubectl wait --for=condition=ready rayservice/grounding-dino-service -n "$NAMESPACE" --timeout="${TIMEOUT_SECONDS}s"

echo "Waiting for Ray cluster pods to be running (timeout: ${TIMEOUT_SECONDS}s)..."
kubectl wait --for=condition=ready pod -l rayCluster=grounding-dino-service -n "$NAMESPACE" --timeout="${TIMEOUT_SECONDS}s"

echo "Grounding DINO service deployment status:"
kubectl get rayservice grounding-dino-service -n "$NAMESPACE"
kubectl get pods -l rayCluster=grounding-dino-service -n "$NAMESPACE"

echo "Setting up port forwarding..."
kubectl port-forward svc/grounding-dino-service-svc 8000:8000 -n "$NAMESPACE" &
PORT_FORWARD_PID=$!

sleep 5

echo "Grounding DINO service deployed successfully!"
echo ""
echo "Service is available at: http://localhost:8000"
echo "Health check: curl http://localhost:8000/health"
echo ""
echo "To stop port forwarding: kill $PORT_FORWARD_PID"
echo "To run tests: ./grounding-dino-service/tests/run_tests.sh"
echo ""
echo "Useful commands:"
echo "  kubectl logs -f deployment/grounding-dino-service-raycluster-head -n $NAMESPACE"
echo "  kubectl get rayservice -n $NAMESPACE"
echo "  kubectl describe rayservice grounding-dino-service -n $NAMESPACE"
