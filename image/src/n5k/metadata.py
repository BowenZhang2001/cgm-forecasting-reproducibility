"""Nutrition5k dish metadata parsing.

Each row of dish_metadata_cafe{1,2}.csv is
    dish_id, total_calories, total_mass, total_fat, total_carb, total_protein,
followed by 7-field groups per ingredient:
    ingr_id, ingr_name, grams, calories, fat, carb, protein
"""
from pathlib import Path

import pandas as pd

TARGETS = ["calories", "carb", "protein", "fat"]
INGR_FIELDS = ["ingr_id", "ingr_name", "grams", "calories", "fat", "carb", "protein"]


def read_dish_metadata(meta_dir):
    dishes, ingredients = [], []
    for cafe in (1, 2):
        path = Path(meta_dir) / f"dish_metadata_cafe{cafe}.csv"
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            f = line.split(",")
            dish_id = f[0]
            dishes.append(dict(
                dish_id=dish_id, cafe=cafe,
                calories=float(f[1]), mass=float(f[2]), fat=float(f[3]),
                carb=float(f[4]), protein=float(f[5]),
                n_fields=len(f),
            ))
            rest = f[6:]
            for i in range(0, len(rest) - len(rest) % 7, 7):
                g = dict(zip(INGR_FIELDS, rest[i:i + 7]))
                for k in ("grams", "calories", "fat", "carb", "protein"):
                    g[k] = float(g[k])
                g["dish_id"] = dish_id
                ingredients.append(g)
    return pd.DataFrame(dishes), pd.DataFrame(ingredients)


def read_ids(path):
    return [s.strip() for s in Path(path).read_text().splitlines() if s.strip()]
