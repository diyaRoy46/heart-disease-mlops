# Screenshots for the final report

Capture these after running the corresponding stage and drop them here (the
report in `docs/report.md` references these filenames):

| File | How to capture |
|---|---|
| `mlflow-runs.png` | `make mlflow-ui` → experiment *heart-disease-classification* run table |
| `mlflow-artifacts.png` | any run → Artifacts → `plots/roc_curve.png` |
| `ci-pipeline.png` | GitHub → Actions → a green CI run (all four jobs) |
| `grafana-dashboard.png` | `make compose-up`, send a few `/predict` requests, open the **Heart Disease API** dashboard on :3000 |
| `prometheus-targets.png` | :9090 → Status → Targets (`heart-disease-api` **UP**) |
| `kubectl-get-all.png` | `kubectl get all` after `make k8s-apply` |
| `k8s-curl.png` | `curl http://<EXTERNAL-IP>/health` against the LoadBalancer |
