install:
	pip install -r requirements.txt

build:
	python3 setup.py sdist bdist_wheel

clean:
	rm -r */__pycache__

clean-build:
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info