.PHONY: help venv data figures train service test docker-build docker-run clean

VENV := .venv
PYTHON := $(VENV)/bin/python
ifeq ($(OS),Windows_NT)
    PYTHON := $(VENV)/Scripts/python.exe
endif

help:
	@echo "KESSLER-SENTINEL Operational Commands:"
	@echo "  make venv         - Create Python virtual environment and install dependencies"
	@echo "  make data         - Generate deterministic LEO conjunction dataset"
	@echo "  make train        - Execute training, threshold optimization and export model artifact"
	@echo "  make figures      - Generate high-resolution technical EDA & performance figures"
	@echo "  make service      - Launch FastAPI conjunction triage microservice locally"
	@echo "  make test         - Run real-time triage scenario tests against inference service"
	@echo "  make docker-build - Build production Docker container image"
	@echo "  make docker-run   - Run containerized microservice on port 9696"

venv:
	python -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) src/build_cdm_dataset.py --output data/leo_conjunction_cdm_events.csv --samples 18000

figures:
	$(PYTHON) src/generate_figures.py
	$(PYTHON) src/render_architecture_diagram.py

train:
	$(PYTHON) train.py --data-path data/leo_conjunction_cdm_events.csv --output-model model_kessler_sentinel_v1.bin

service:
	$(PYTHON) -m uvicorn predict:app --host 0.0.0.0 --port 9696

test:
	$(PYTHON) -m pytest -v
	$(PYTHON) test_service.py --url http://127.0.0.1:9696

docker-build:
	docker build -t kessler-sentinel-triage:v1 .

docker-run:
	docker run -it --rm -p 9696:9696 --name kessler-triage-container kessler-sentinel-triage:v1

clean:
	rm -rf __pycache__ src/__pycache__ .pytest_cache
