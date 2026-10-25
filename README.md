# Iris ML — MLOps end-to-end

Python ML model -> Docker -> AWS (ECS Fargate) -> GitHub Actions CI/CD -> Terraform -> Monitoring/Logging

```
app/train.py            scikit-learn training (fails build if accuracy < 0.9)
app/main.py             FastAPI: /predict /health /metrics, JSON logs
tests/                  pytest
Dockerfile              trains at build time, non-root, healthcheck
terraform/              ECR, ECS Fargate, ALB, autoscaling, CloudWatch (logs, alarms, dashboard)
.github/workflows/      test -> build -> push -> terraform apply -> smoke test
docker-compose.monitoring.yml   local API + Prometheus + Grafana
```

## 1. Local run
```bash
pip install -r requirements-dev.txt
pytest -q
python -m app.train model
uvicorn app.main:app --reload
curl -X POST localhost:8000/predict -H 'content-type: application/json' \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

## 2. Docker
```bash
docker build -t iris-ml .
docker run -p 8000:8000 iris-ml
```

## 3. Local monitoring (Prometheus + Grafana)
```bash
docker compose -f docker-compose.monitoring.yml up --build
# Prometheus http://localhost:9090   Grafana http://localhost:3000 (admin/admin)
# In Grafana add Prometheus datasource: http://prometheus:9090
```
Useful queries: `rate(http_requests_total[1m])`,
`histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))`, `predictions_total`.

## 4. One-time AWS setup
1. Create an S3 bucket for Terraform state (versioning on).
2. Create an IAM OIDC provider for `token.actions.githubusercontent.com` and an IAM role
   trusted for your repo (`repo:<org>/<repo>:ref:refs/heads/main`) with permissions for
   ECR, ECS, ELB, EC2 (SG/describe), IAM (role create/pass), CloudWatch, SNS, Application Auto Scaling, S3 (state bucket).
3. In GitHub repo settings:
   - Secret `AWS_ROLE_ARN`
   - Variables `TF_STATE_BUCKET`, `ALERT_EMAIL`
   - Create an environment named `production` (optionally add required reviewers)

## 5. Deploy
Push to `main`. The pipeline: tests -> terraform fmt/validate -> docker smoke test ->
ensure ECR -> push image tagged with commit SHA -> `terraform apply` -> wait for ECS stable -> smoke test ALB.
Rollback is automatic if the new tasks fail health checks (ECS deployment circuit breaker).

Manual Terraform:
```bash
cd terraform
terraform init -backend-config="bucket=<bucket>" -backend-config="key=iris-ml/terraform.tfstate" -backend-config="region=ap-south-1"
terraform apply -target=aws_ecr_repository.app
# push an image to the ECR URL, then:
terraform apply -var="image_tag=<tag>" -var="alert_email=you@example.com"
terraform destroy   # clean up to avoid charges
```

## 6. Monitoring & logging on AWS
- Logs: JSON lines -> CloudWatch Logs `/ecs/iris-ml`. Query in Logs Insights:
  `fields @timestamp, method, path, status, duration_ms | filter status >= 500 | sort @timestamp desc`
- Metric filter converts `level=ERROR` logs into a custom metric.
- Alarms (SNS email): app errors, ALB 5xx, unhealthy hosts, high CPU.
- Dashboard: requests/5xx, p95 latency, CPU/memory, recent errors (see `dashboard_url` output).
- Container Insights is enabled on the ECS cluster.

## Azure instead of AWS?
Same Docker image works. Swap the Terraform resources for `azurerm_container_registry` +
`azurerm_container_app` (+ Log Analytics workspace + Azure Monitor alerts), and use
`azure/login` (OIDC) + `az acr build` in the workflow.
