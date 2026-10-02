import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
DATA_DIR = HERE / "data" / "raw" / "Cattle side and back view images"

measurements = pd.read_excel(DATA_DIR / "measurements.xlsx")
print(measurements.head())

