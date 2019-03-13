build:
	python3 setup.py sdist bdist_wheel

clean-build:
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info