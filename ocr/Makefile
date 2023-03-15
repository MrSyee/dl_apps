run-server:
	PYTHONPATH=server uvicorn app:app --host 0.0.0.0 --port 8888 --reload

run-frontend:
	PYTHONPATH=frontend streamlit run frontend/frontend.py


# dev
format:
	black .
	isort .