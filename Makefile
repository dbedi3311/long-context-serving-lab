.PHONY: setup-mac models run bench

setup-mac:
	pip install -r requirements-mac.txt

models:
	python scripts/download_models.py --path models

run:
	uvicorn server.app:app --reload

bench:
	python bench/bench_kv_paging.py
	python bench/bench_latency.py
	python bench/bench_specdecode.py
