The goal is to deploy a variety of ML models for vision and language inference using Ray Serve on Kubernetes.

This is part of a compilation of sketches of the architecture, before starting to build.

---

## Production Guide: Deploying Ray Serve on Kubernetes with Datadog Integration

### Overview

This guide covers a production-grade setup for deploying Ray Serve models to Kubernetes, with an emphasis on multi-user isolation and observability. We will integrate Ray Serve with Datadog for logging, metrics, and tracing, and provide a CLI-driven deployment workflow using templated Kubernetes manifests and shell scripts. The solution ensures each user’s deployment is isolated by name prefix and can target specific node pools via node selectors.

### 1. Integrating Ray Serve with Datadog

Logs forwarding: Configure your Ray Serve deployments to log to stdout/stderr only – this is a Kubernetes best practice, as the container runtime automatically collects these streams on each node. With the Datadog Agent installed as a DaemonSet on the cluster, you can enable log collection (e.g., via containerCollectAll: true in the agent config). The DaemonSet will schedule an agent on every node, and each agent will automatically pick up logs from all containers on that node. In practice, this means any logs your Ray Serve model prints to stdout/stderr will be forwarded to Datadog without additional configuration. For clarity, you can also add a Datadog logs annotation to tag these logs (for example, to set a custom log service or source):

```
metadata:
  annotations:
    ad.datadoghq.com/ray.logs: '[{"source": "ray", "service": "${SERVICE_NAME}"}]'
```

This optional annotation (applied to the Pod template) tells the Datadog agent to treat logs from the container as coming from the “ray” source and to tag them with a service name (e.g. the model or deployment name). If your Datadog agent is already configured to collect all container logs, the annotation is not strictly required, but it can help in categorizing logs.

Metrics collection: Ray exposes internal metrics in Prometheus (OpenMetrics) format. By default, Ray does not persist metrics itself – you must scrape or push them to a monitoring system. To expose metrics, ensure Ray’s metrics server is enabled on each node. In a Kubernetes deployment, you can pass --metrics-export-port to ray start or set the rayStartParams.metrics-export-port in a Ray cluster config (for KubeRay). For example, you might choose port 8080 for metrics export. All Ray metrics use the ray. prefix, including Serve metrics and any custom metrics you define. Key Serve metrics include request throughput and latency – for instance, ray.serve.http_request_latency measures end-to-end HTTP inference latency in milliseconds, and ray.serve.deployment.processing_latency measures the time spent processing requests inside the replica. These provide insight into your model’s performance (e.g. p95 latency). Datadog’s Ray integration supports scraping these metrics via an OpenMetrics endpoint.

To have the Datadog Agent collect Ray’s Prometheus-compatible metrics, use Datadog’s autodiscovery with an OpenMetrics check. Annotate the Ray Serve pod with the integration config pointing to the metrics port. For example:

```
metadata:
  annotations:
    ad.datadoghq.com/ray.checks: |-
      {
        "ray": {
          "instances": [
            { "openmetrics_endpoint": "http://%%host%%:8080" }
          ]
        }
      }
```

In this JSON, "ray" refers to Datadog’s built-in Ray integration and openmetrics_endpoint tells the agent to scrape the pod’s own IP on port 8080. The %%host%% token is automatically replaced by the agent with the pod’s IP address. Important: Ensure the container’s name matches the key in the annotations. In our case, we name the container "ray", so we use ad.datadoghq.com/ray.checks as the annotation key. If you use a different container name, the annotation key must change accordingly. With this in place, the Datadog agent will poll the Ray metrics endpoint and forward metrics (including inference timing, error rates, etc.) to Datadog. Datadog’s out-of-the-box Ray dashboard can then visualize these metrics.

Tip: Ray allows exporting custom application-level metrics (e.g., business-specific counters) via its ray.util.metrics API. Any custom metrics you create with the ray. prefix can also be collected by Datadog. You can list them under extra_metrics in the integration config if needed, but by default the Ray integration will handle standard Ray/Serve metrics.

Tracing (optional): Ray Serve can integrate with OpenTelemetry for distributed tracing. In fact, each request handled by Ray Serve automatically generates an OpenTelemetry trace by default (spanning from the HTTP/gRPC ingress through the Ray Serve pipeline). These traces can be very useful for debugging and performance analysis. For a “cheap” tracing solution (i.e. minimal overhead), you have a few options:

* Ray’s built-in tracing to logs: By default, Ray will log trace spans in the Ray logs (visible via the Ray dashboard or stdout). This requires no external infrastructure – you can increase the sampling rate or enable tracing in the Serve deployment config (e.g., set tracing_config.enabled=True and sampling_ratio=1.0 for full sampling) if using the Serve API. The traces will then appear in the logs which are already forwarded to Datadog, allowing you to inspect end-to-end request flows in the log stream.
* OpenTelemetry exporter: For more structured tracing, you can instrument the application with the OpenTelemetry SDK. Ray supports configuring an OTLP exporter to send traces to an external backend. For example, you could deploy a lightweight OpenTelemetry Collector or configure the Datadog Agent to accept OTLP trace data, and then point Ray’s tracer there. In code, this might involve installing opentelemetry-exporter-otlp and setting an exporter endpoint (Datadog can accept OTLP over gRPC on port 4317 or HTTP on 4318 when configured). The Ray Serve tracing guide (experimental) provides examples of sending traces to Jaeger or other backends, which can be adapted to Datadog. Alternatively, you could use Datadog’s APM client (e.g., ddtrace for Python) within your Ray Serve deployment, though integrating that with Ray’s multi-actor architecture may require additional configuration.
* Sampling and overhead: To keep tracing “cheap,” you can enable a low sampling rate (e.g., sample a small percentage of requests) or only trace certain endpoints. Ray’s tracing_config allows adjusting the sampling_ratio. Even with tracing on, the overhead is modest if sampling is low, as Ray’s integration is optimized for minimal performance impact when exporting spans.

In summary, you have the option to get basic distributed tracing visibility with minimal changes (just enabling Ray’s OpenTelemetry support and using the Datadog Agent or Collector), providing you with per-request latency breakdowns in addition to the aggregate metrics.

### 2. CLI-Driven Deployment System (kubectl + shell)

To support multi-user deployments of Ray Serve models on a shared cluster, we provide a CLI tool (deploy_ray_service.sh) and templated Kubernetes manifests. The goal is to let users deploy their own instance of a model serving service with a single command, while automatically customizing the Kubernetes resources for isolation and scheduling needs.

Key features:
* Username prefix for resources: Each Deployment and Service will be prefixed with the current UNIX username. For example, if user alice deploys a service named “clip-service”, the actual Kubernetes Deployment and Service will be named alice-clip-service. This prevents naming collisions in a multi-user environment and makes it easy to identify ownership of services. The prefixing is done automatically by the script (using $(whoami) or the $USER environment variable).
* Node selector per deployment: Users can target specific node pools (e.g., GPU nodes, high-memory nodes) by providing a node selector. The script accepts a --node-selector key=value argument. The provided key/value will be injected into the pod spec’s nodeSelector, ensuring the Ray Serve pod runs on nodes with that label. For example, --node-selector accelerator=nvidia could target GPU nodes. If no selector is provided, the service will schedule on any available node.
* Flexible annotations: The deployment template includes placeholders for annotations (used for Datadog integration or other purposes). The script can inject annotation key-value pairs provided via arguments (e.g., --annotation key=value). This allows enabling Datadog Autodiscovery for metrics as shown above, or adding custom metadata. In our template, we include the Datadog metrics and logs annotations by default for convenience, but these can be extended or overridden as needed.
* Environment variable templating: We use a lightweight templating approach with environment variable substitution (via the Unix envsubst tool). The manifest files contain variables like $USERNAME, $SERVICE_NAME, $NODE_LABEL_KEY, etc., which the script replaces with actual values at deploy time. This avoids the complexity of maintaining separate files per user or requiring a heavier templating solution. (For more complex configurations, tools like Helm or Kustomize could be used, but envsubst keeps things simple and portable.)


Kubernetes manifest template: Below is an example Kubernetes manifest (ray_serve_deployment.yaml) with placeholders. It includes a Deployment for a Ray Serve instance and a corresponding Service. This template will be processed by the shell script to fill in the username, service name, node selector, and annotations.

```
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${USERNAME}-${SERVICE_NAME}
  labels:
    app: ${USERNAME}-${SERVICE_NAME}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ${USERNAME}-${SERVICE_NAME}
  template:
    metadata:
      labels:
        app: ${USERNAME}-${SERVICE_NAME}
      annotations:
        ad.datadoghq.com/ray.checks: |-
          {
            "ray": {
              "instances": [
                { "openmetrics_endpoint": "http://%%host%%:8080" }
              ]
            }
          }
        ad.datadoghq.com/ray.logs: '[{"source": "ray", "service": "${SERVICE_NAME}"}]'
        # Additional user-provided annotations can be inserted below (if any)
        # NODE_SELECTOR_PLACEHOLDER
    spec:
      # If a node selector is specified, it will be inserted here by the script.
      containers:
      - name: ray
        image: <YOUR_RAY_SERVE_IMAGE>:latest
        ports:
        - containerPort: 8000   # Ray Serve HTTP port (defaults to 8000)
        - containerPort: 8080   # Prometheus metrics export port
        env:
        - name: RAY_START_PARAMS
          value: "--head --dashboard-host=0.0.0.0 --metrics-export-port=8080"
        command: ["/bin/bash"]
        args:
        - "-c"
        - |
          # Start Ray head node and Ray Serve application
          ray start $RAY_START_PARAMS && \
          python serve_app.py
          # serve_app.py should contain your Ray Serve deployment initialization (e.g., serve.run or deployment definitions)
---
apiVersion: v1
kind: Service
metadata:
  name: ${USERNAME}-${SERVICE_NAME}
  labels:
    app: ${USERNAME}-${SERVICE_NAME}
spec:
  type: ClusterIP
  selector:
    app: ${USERNAME}-${SERVICE_NAME}
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: http
```

In this template:
* Name prefix: Both Deployment and Service metadata.name use ${USERNAME}-${SERVICE_NAME}. The same composite label is applied for consistent service selection.
* Datadog annotations: The ad.datadoghq.com/ray.checks and ad.datadoghq.com/ray.logs annotations are included on the pod template. These will be rendered with the appropriate service name and instruct the Datadog agent to scrape metrics from port 8080 and collect logs. The container is named "ray" to match the annotation key.
* Node selector placeholder: We have a commented placeholder # NODE_SELECTOR_PLACEHOLDER. The script will replace this with a nodeSelector section if the user provided a node selector, or remove it if not. By default (if no selector), the pod can run on any node.
* Ray Serve startup: The container command starts Ray (as head node) with metrics enabled (--metrics-export-port=8080 and --dashboard-host=0.0.0.0 to allow metrics scraping from any interface) and then runs a user script serve_app.py. In practice, serve_app.py would call ray.init(address="auto") and define your Ray Serve deployments (or use serve.run) to start the model server. This approach encapsulates the entire Ray Serve instance in one pod for simplicity. In more advanced setups, you could use the Ray Operator (CRD) to launch a multi-pod Ray cluster, but the one-pod approach is often sufficient for model serving and easier to manage per user.

Deployment shell script: The following shell script (deploy_ray_service.sh) automates the templating and kubectl deployment. It takes optional flags for node selector and annotations, and a required argument for the base service name. It derives the username from the environment. The script ensures the manifest is rendered with the correct values and then applies it to the cluster:

```
#!/bin/bash
set -euo pipefail

# Usage message
usage() {
  echo "Usage: $0 [--node-selector key=value] [--annotation key=value] <service_name>" >&2
  exit 1
}

# Default variables
NODE_LABEL_KEY=""
NODE_LABEL_VALUE=""
EXTRA_ANNOTATION_KEY=""
EXTRA_ANNOTATION_VALUE=""

# Parse arguments
if [[ $# -lt 1 ]]; then
  usage
fi
while [[ $# -gt 0 ]]; do
  case "$1" in
    --node-selector|-n)
      [[ -z "${2:-}" || "$2" != *=* ]] && usage
      NODE_LABEL_KEY="${2%%=*}"
      NODE_LABEL_VALUE="${2#*=}"
      shift 2
      ;;
    --annotation|-a)
      [[ -z "${2:-}" || "$2" != *=* ]] && usage
      EXTRA_ANNOTATION_KEY="${2%%=*}"
      EXTRA_ANNOTATION_VALUE="${2#*=}"
      shift 2
      ;;
    --help|-h)
      usage
      ;;
    *)
      # Remaining argument is the service base name
      SERVICE_NAME="$1"
      shift
      ;;
  esac
done

[[ -z "${SERVICE_NAME:-}" ]] && usage

# Determine username (for prefix). Use $USER if set, otherwise whoami.
USERNAME="${USER:-$(whoami)}"

# Export variables for envsubst
export USERNAME SERVICE_NAME

# Copy template to a temp file for manipulation
TEMPLATE_FILE="ray_serve_deployment.yaml"
TMP_FILE="$(mktemp)"
cp "$TEMPLATE_FILE" "$TMP_FILE"

# Insert node selector if provided
if [[ -n "$NODE_LABEL_KEY" && -n "$NODE_LABEL_VALUE" ]]; then
  # Replace placeholder with nodeSelector YAML
  sed -i "s|# NODE_SELECTOR_PLACEHOLDER|nodeSelector:\\n        ${NODE_LABEL_KEY}: ${NODE_LABEL_VALUE}|g" "$TMP_FILE"
else
  # Remove the placeholder comment line if no node selector
  sed -i "/# NODE_SELECTOR_PLACEHOLDER/d" "$TMP_FILE"
fi

# Insert extra annotation if provided (on the pod template metadata)
if [[ -n "$EXTRA_ANNOTATION_KEY" && -n "$EXTRA_ANNOTATION_VALUE" ]]; then
  # Format the annotation as YAML (assuming string value for simplicity)
  sed -i "/# Additional user-provided annotations/a\        ${EXTRA_ANNOTATION_KEY}: \"${EXTRA_ANNOTATION_VALUE}\"" "$TMP_FILE"
fi

# Apply the manifest with variables substituted
envsubst < "$TMP_FILE" | kubectl apply -f -
rm -f "$TMP_FILE"

echo "Deployment ${USERNAME}-${SERVICE_NAME} has been applied."
```

Script explanation:

* It uses getopts-like manual parsing to handle --node-selector and --annotation. The node selector expects a value of the form key=value. The annotation expects key=value as well (for a single extra annotation; this can be extended to multiple if needed by repeating the flag).
* The script exports USERNAME and SERVICE_NAME so that envsubst can substitute those in the YAML. We copy the template to a temporary file and then use sed to handle the conditional parts:
* If a node selector was provided, the script finds the # NODE_SELECTOR_PLACEHOLDER line and replaces it with a nodeSelector: YAML entry (with proper indentation and the given key/value). If no selector was given, it simply removes the placeholder line, resulting in no nodeSelector in the final manifest.
* If an extra annotation was provided via --annotation, the script appends that annotation under the existing annotations section. It uses the marker comment # Additional user-provided annotations to know where to insert the line. The inserted line is indented to align under metadata.annotations. (Multiple annotations could be handled by multiple -a flags; one could modify the script to accumulate them in a loop and insert each, but for simplicity we show a single annotation insertion. In practice, Datadog metrics and logs config are already in the template, so additional annotations might not be needed often.)
* Finally, the script runs envsubst on the (now customized) temp file and pipes it to kubectl apply. This will create/update the Deployment and Service in Kubernetes. The temp file is then removed.

Usage example: Suppose user alice wants to deploy a CLIP model serving service on GPU nodes. The image for the service is already built into the template (or could be specified via another variable if needed). She can run:

`./deploy_ray_service.sh --node-selector accelerator=nvidia-gpu clip-service`

The script will detect her username (alice), set SERVICE_NAME=clip-service, and inject the node selector accelerator: nvidia-gpu into the pod spec. The Kubernetes resources created will be Deployment/alice-clip-service and Service/alice-clip-service. The pod will only schedule on nodes labeled accelerator=nvidia-gpu. The Datadog agent (running on those nodes) will autodiscover the metrics endpoint on port 8080 and start sending Ray Serve metrics (like request latency, throughput, errors) to Datadog. Alice’s logs from the model (stdout/stderr) will be collected by the agent on that node and viewable in Datadog’s log explorer. If tracing is enabled in the application, she can correlate logs and metrics with trace IDs (Datadog will automatically attempt to correlate if the trace and log contexts are linked, or she can search logs for request IDs output by the tracing). Meanwhile, another user bob can run the same script for his own model (e.g., ./deploy_ray_service.sh --node-selector nodeType=highmem summarizer) and it will create Deployment/bob-summarizer without affecting Alice’s service.

Multi-user robustness: By prefixing resource names with the username, collisions are avoided even if two users deploy services with the same base name. The use of labels and selectors tied to those names ensures each Service routes to the correct pods. The templating approach confines each user’s custom values to their deployment only. Additionally, Kubernetes RBAC can be configured (outside the scope of this guide) so that users can only manage their own prefixed resources if needed. The shell script approach, while simple, is effective for a shared cluster: it avoids requiring each user to manually edit YAML and reduces errors by programmatically inserting the correct values.

Additional production considerations:

* Namespace isolation: In this example, all deployments go to the default or current namespace. For stricter isolation, you could assign each user a separate Kubernetes namespace. The script could then also take a --namespace parameter or derive one from the username. This would further prevent any naming conflicts and allow resource quotas per user.
* Cleanup and updates: The script uses kubectl apply, which handles both creation and in-place updates. If a user re-runs the deployment with the same name, it will update the existing Deployment (e.g., new image, new config) without downtime (Kubernetes will roll out changes). To remove a deployment, the user can run kubectl delete deployment,service ${USERNAME}-${SERVICE_NAME} or a separate teardown script could be provided.
* Scaling and replicas: The template sets replicas: 1 for simplicity (each Ray Serve deployment is a single pod running a Ray head and worker in one). If you need to scale out a Ray Serve application to multiple pods, you would typically use Ray’s cluster mode (with a head and multiple workers) rather than replicating the same pod, due to how Ray clustering works. However, you could run multiple replicas of the same pod behind a Service (they would be independent Ray instances, not a coordinated cluster, but could still serve traffic if they each run the model). For true horizontal scaling of a single Serve deployment, consider using the Ray Operator (KubeRay) to manage a Ray cluster – that would be a more complex setup, beyond this guide’s scope.
* Datadog APM integration: If you require deep tracing integration with Datadog (APM), you might instrument the model code with Datadog’s tracer. This could involve adding the ddtrace library and setting environment variables like DD_SERVICE, DD_ENV, DD_VERSION to identify the service, and pointing the tracer to the Datadog Agent. The agent itself (if running on the node) can accept traces on port 8126. Another approach is using OTLP as mentioned. Since Ray Serve requests may span multiple processes (ingress, controller, replica), ensuring a single trace ID flows through might require context propagation. Ray’s OpenTelemetry support handles context propagation between tasks/actors, so leveraging that with Datadog’s OTLP ingest is a clean solution.

### 3. Datadog Metrics and Monitoring Configuration

With the above deployment approach, each Ray Serve service will automatically feed metrics and logs into Datadog:
* Logs in Datadog: Once logs are in Datadog, you can create log-based metrics or monitors (e.g., alert on error messages or high latency warnings in logs). Ensure the logs have any relevant context (for example, logging the request ID or model version can be useful).
* Metrics dashboard: The Datadog Ray integration provides a pre-built dashboard. You should see metrics like ray.serve.num_http_requests (requests per second), ray.serve.http_request_latency.sum/count (for latency distributions), and resource usage metrics (CPU, memory of the Ray process, etc.). You can filter these by service or deployment if the Datadog integration tags them accordingly. By default, the Datadog agent will tag metrics with the pod’s tags (including kube_deployment and service names), so Alice’s metrics might carry a tag like kube_deployment:alice-clip-service.
* Alerts on latency: Using the collected latency metrics, you can set up monitors in Datadog. For example, monitor the histogram ray.serve.http_request_latency.bucket to alert if the 95th percentile exceeds a threshold, or use ray.serve.http_request_latency.sum/ray.serve.http_request_latency.count to alert on the average latency. These metrics are emitted by Ray Serve’s HTTP proxy, so they capture the end-to-end time of requests through the serving system.
* Tracing visualization: If you exported traces, you can use Datadog APM’s Trace view to see individual requests. This will show spans for the Ray Serve request; for instance, you might see a parent span for the HTTP request and child spans for internal processing (Ray tasks/actor invocations) if configured. This can help pinpoint where latency is introduced (e.g., in data preprocessing vs. model inference). Because Ray’s integration is OpenTelemetry-based, it’s compatible with Datadog’s tracing after some setup.

Finally, all these pieces (logging, metrics, tracing) come together to give a comprehensive observability picture. By following this setup guide, you achieve robust, repeatable deployments for each user’s model service and integrate them with Datadog’s monitoring platform for production-grade insights.

Sources:
* Datadog Kubernetes logging: Each node’s Datadog agent collects container logs for pods on that node. It’s best practice to write application logs to stdout/stderr for easy collection.
* Ray Serve metrics exposure: Enable Ray’s OpenMetrics endpoint (e.g. via --metrics-export-port=8080 on ray start) to collect metrics. Datadog’s Ray integration can scrape this endpoint, capturing metrics like ray.serve.http_request_latency (end-to-end inference latency in milliseconds).
* Datadog Autodiscovery annotations: Use ad.datadoghq.com/<container>.checks with the Ray integration to configure metrics scraping on the pod. Ensure the <container> name in the annotation matches your pod’s container name.
* Ray Serve tracing: Ray integrates with OpenTelemetry for distributed tracing. By default each Serve request can produce a trace spanning the request’s lifecycle, which can be exported or logged for lightweight tracing. Anyscale’s tracing guide provides examples of configuring Ray Serve tracing and exporting to external backends.
