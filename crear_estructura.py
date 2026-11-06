import os
from pathlib import Path
import shutil

# 1. Definir la raíz y las rutas principales
ROOT = Path(".")
DATA_DIR = ROOT / "data"
NOTEBOOKS_DIR = ROOT / "notebooks"
PKG_DIR = ROOT / "datascience_toolkit"

# Submódulos del nuevo paquete
MODULES = [
    "eda",
    "preprocessing",
    "unsupervised",
    "supervised",
    "visualization",
    "marketing",
]

# 2. Crear carpetas data/ y notebooks/
DATA_DIR.mkdir(exist_ok=True)
NOTEBOOKS_DIR.mkdir(exist_ok=True)

# 3. Crear el esqueleto del paquete datascience_toolkit
PKG_DIR.mkdir(exist_ok=True)
(PKG_DIR / "__init__.py").touch(exist_ok=True)

for mod in MODULES:
  mod_path = PKG_DIR / mod
  mod_path.mkdir(exist_ok=True)
  (mod_path / "__init__.py").touch(exist_ok=True)

# 4. Mover archivos CSV existentes a data/
for file in ROOT.glob("*.csv"):
  shutil.move(str(file), str(DATA_DIR / file.name))
  print(f"📦 Movido a data/: {file.name}")

# 5. Mover notebooks existentes a notebooks/ (excepto si decides dejar alguno)
for file in ROOT.glob("*.ipynb"):
  shutil.move(str(file), str(NOTEBOOKS_DIR / file.name))
  print(f"📓 Movido a notebooks/: {file.name}")

# 6. Crear un pyproject.toml base para instalación en modo editable
pyproject_content = """[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "datascience_toolkit"
version = "0.1.0"
description = "Modular Data Science toolkit for analytics, clustering and marketing activation"
readme = "README.md"
authors = [{ name = "Victor" }]
requires-python = ">=3.9"
dependencies = [
    "numpy",
    "pandas",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "plotly"
]

[project.optional-dependencies]
advanced = [
    "umap-learn",
    "boto3",
    "awswrangler"
]
"""

pyproject_file = ROOT / "pyproject.toml"
if not pyproject_file.exists():
  pyproject_file.write_text(pyproject_content, encoding="utf-8")
  print("⚙️ Creado: pyproject.toml")

print("\n✅ Estructura base inicializada con éxito.")