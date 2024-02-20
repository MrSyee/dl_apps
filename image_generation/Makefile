setup:
	pip install -r requirements.txt

# dev
setup-dev:
	pip install -r ../requirements-dev.txt

format:
	black .
	isort .

run:
	CUDA_VISIBLE_DEVICES=6 python sketch_to_image.py