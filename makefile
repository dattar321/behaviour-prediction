.PHONY: setup
setup:
	pip install -e .

.PHONY: template
template:
	python template.py
