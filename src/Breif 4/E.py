import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
DATA_DIR = HERE / "data" / "raw" / "Cattle side and back view images"

measurements = pd.read_excel(DATA_DIR / "measurements.xlsx")
print(measurements.head())

animal_num = 1
row = measurements[measurements["Num"] == animal_num]
weight = row["Body weight (kg)"].values[0]

image_path = DATA_DIR / "side view" / f"{animal_num}.png"
img = Image.open(image_path)

plt.imshow(img)
plt.title(f"Cow #{animal_num} — recorded weight: {weight} kg")
plt.axis("off")
plt.show()