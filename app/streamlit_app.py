import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

from pipelines.lotomania_ai_pipeline import (
    generate_features,
    predict_top_50,
    evaluate,
    gerar_10_combinacoes
)

BASE_DIR = Path(__file__).resolve().parent.parent
...