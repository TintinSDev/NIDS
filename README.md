# SENTINEL — AI Network Intrusion Detection System (NIDS)

**SENTINEL** is an AI-powered Network Intrusion Detection System (NIDS) that monitors network traffic, extracts behavioral flow features, detects suspicious and malicious activity, and provides explainable security alerts through a Security Operations Center (SOC) dashboard.

It combines real-time packet capture using **Scapy**, 52-feature bidirectional flow aggregation, a calibrated multi-model detection engine, SHAP-based explainability, PostgreSQL persistence, and a FastAPI-powered monitoring dashboard.

## Architecture

SENTINEL consists of four main components:

### 1. Live Capture Worker
**`src/capture/live_capture.py`**

- Managed through `systemd` using `nids-capture.service`.
- Configuration is maintained in `/etc/default/nids-capture`.
- Captures live network packets through Scapy on the `wlo1` interface.
- Aggregates packets into bidirectional network flows using `FlowAccumulator`.
- Extracts the 52 features defined in `features_v1.json`.
- Flushes completed flow records in batches to the detection API through `POST /analyze/flows`.

### 2. Detection & API Service
**`src/api/main.py` — `nids_api`**

- Processes incoming live network flows and uploaded CSV datasets.
- Supports offline analysis through `POST /analyze/upload`.
- Uses calibrated machine learning artifacts stored in `artifacts/models/v1`.
- Combines multiple detection models:
  - XGBoost
  - Random Forest
  - Isolation Forest
  - Multiclass Attack Classifier
- Classifies network activity into three detection tiers:
  - `benign`
  - `suspicious`
  - `confirmed_malicious`
- Generates top-5 SHAP feature attributions for security alerts to explain the factors contributing to detection decisions.

### 3. Data Persistence
**`nids_postgres`**

- Uses PostgreSQL 16 for persistent storage.
- Stores analyzed network flows in the `flows` table.
- Distinguishes between live and offline traffic using the `source` field:
  - `live`
  - `offline`
- Stores generated security alerts in the `alerts` table.
- Stores supporting detection evidence in the `alert_evidence` table.

### 4. SOC Dashboard
**`src/api/static/index.html`**

The web-based Security Operations Center dashboard is served directly by FastAPI.

Access it at:

[http://localhost:8000/](http://localhost:8000/)

The dashboard provides five main monitoring sections:

- **Overview** — High-level network monitoring and detection statistics.
- **Alerts** — Security alerts, SHAP evidence drawer, and alert triage actions.
- **Traffic** — Network flow monitoring and traffic analysis.
- **Models** — Machine learning detection model information.
- **Capture** — Live packet capture monitoring.

The dashboard uses live polling to retrieve updated monitoring information from the API.

---
---

## Key API Endpoints

SENTINEL exposes REST API endpoints for network traffic ingestion, intrusion detection, monitoring, and security alert management.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the SENTINEL SOC Dashboard UI. |
| `GET` | `/health` | Checks API health and service availability. |
| `GET` | `/stats` | Returns flow source distribution and alert counts by severity and classification. |
| `GET` | `/alerts` | Retrieves the security alert feed. |
| `GET` | `/alerts/{id}` | Retrieves alert details, SHAP explanations, and detector evidence. |
| `PATCH` | `/alerts/{id}/status` | Updates alert status for analyst triage (`acknowledged`, `false_positive`, or `resolved`). |
| `GET` | `/flows` | Retrieves live and offline network flow records. |
| `GET` | `/traffic/summary` | Returns traffic analytics, including top talkers, protocols, and ports. |
| `POST` | `/analyze/flows` | Ingests and analyzes real-time network flows submitted as JSON. |
| `POST` | `/analyze/upload` | Uploads and analyzes network flow datasets in CSV format. |

### Alert Triage Example

Update an alert's status using:

```bash
curl -X PATCH \
  "http://localhost:8000/alerts/1/status?status=acknowledged"
```

### API Health Check

```bash
curl http://localhost:8000/health
```

### Retrieve Security Alerts

```bash
curl http://localhost:8000/alerts
```


## Quick Start

### Prerequisites

- Linux environment
- Python virtual environment (`.venv`)
- Docker Engine
- Docker Compose
- Scapy and the project's Python dependencies
- `systemd` for managing the live capture service

### 1. Start PostgreSQL and FastAPI Containers

Start the application containers:

```bash
docker compose up -d
```

Verify that the API is running:

```bash
curl -s http://localhost:8000/health
```

### 2. Start and Check the Live Capture Service

Enable and start the network capture service:

```bash
sudo systemctl enable --now nids-capture.service
```

Check the service status:

```bash
sudo systemctl status nids-capture.service --no-pager
```

Inspect the latest service logs:

```bash
sudo journalctl -u nids-capture.service -n 20 --no-pager
```

### 3. Simulate Network Traffic and Verify Detection

Activate the Python virtual environment:

```bash
source .venv/bin/activate
```

Run the attack traffic simulation:

```bash
python3 scripts/simulate_attack.py
```

Check the API statistics to verify the detection pipeline:

```bash
curl -s http://localhost:8000/stats
```

### 4. Open the SOC Dashboard

Open your browser and navigate to:

[http://localhost:8000/](http://localhost:8000/)

Explore the Overview, Alerts, Traffic, Models, and Capture sections to monitor network activity and inspect detection evidence.

---

## Machine Learning & Explainability

SENTINEL uses a multi-model detection architecture combining supervised and unsupervised machine learning techniques.

| Model | Purpose |
|---|---|
| XGBoost | Supervised network intrusion detection |
| Random Forest | Ensemble-based traffic classification |
| Isolation Forest | Unsupervised anomaly detection |
| Multiclass Attack Classifier | Attack category classification |
| SHAP | Explainable detection and feature attribution |

The system uses calibrated model artifacts and a versioned 52-feature schema to maintain consistency between flow extraction and model inference.

SHAP feature attributions provide additional context for investigating detected threats.

---

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Packet Capture | Scapy |
| Feature Engineering | Custom 52-feature Flow Aggregation |
| Machine Learning | XGBoost, Random Forest, Isolation Forest |
| Explainable AI | SHAP |
| Backend API | FastAPI |
| Database | PostgreSQL 16 |
| Containerization | Docker, Docker Compose |
| Service Management | Linux systemd |
| Frontend | HTML, CSS, JavaScript |
| Monitoring | Custom SOC Dashboard |

---

## Project Highlights

- Real-time network packet capture and bidirectional flow aggregation.
- Structured 52-feature network traffic extraction.
- Multi-model intrusion detection combining supervised and unsupervised learning.
- Multiclass attack classification.
- Calibrated detection and threat severity tiers.
- SHAP-based explainable AI for security alert investigation.
- Persistent storage of network flows, alerts, and detection evidence.
- Support for both live traffic analysis and offline CSV-based analysis.
- Centralized SOC dashboard for network monitoring and alert triage.

---

**SENTINEL — Intelligent Network Monitoring, Threat Detection, and Explainable Security Analytics.**