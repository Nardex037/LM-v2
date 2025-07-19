# Makefile
setup:
	pip install -r requirements.txt

train:
	python pipelines/lotomania_ai_pipeline.py

run:
	streamlit run app/streamlit_app.py