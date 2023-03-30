"""Functions for RATA analysis"""
from pathlib import Path

import IPython.display
import pandas as pd
import seaborn as sns


DATA_PATH = Path(__file__).parent.parent / 'data'

FIGURE_PATH = Path(__file__).parent.parent / 'figures'

ODOR_LABEL_LIST = [
u'alcoholic', u'aldehydic', u'alliaceous', u'almond', u'amber', u'animal',
u'anisic', u'apple', u'apricot', u'aromatic', u'balsamic', u'banana',
u'beefy', u'bergamot', u'berry', u'bitter', u'black currant', u'brandy',
u'burnt', u'buttery', u'cabbage', u'camphoreous', u'caramellic', u'cedar',
u'celery', u'chamomile', u'cheesy', u'cherry', u'chocolate', u'cinnamon',
u'citrus', u'clean', u'clove', u'cocoa', u'coconut', u'coffee', u'cognac',
u'cooked', u'cooling', u'cortex', u'coumarinic', u'creamy', u'cucumber',
u'dairy', u'dry', u'earthy', u'ethereal', u'fatty', u'fermented', u'fishy',
u'floral', u'fresh', u'fruit skin', u'fruity', u'garlic', u'gassy',
u'geranium', u'grape', u'grapefruit', u'grassy', u'green', u'hawthorn',
u'hay', u'hazelnut', u'herbal', u'honey', u'hyacinth', u'jasmin', u'juicy',
u'ketonic', u'lactonic', u'lavender', u'leafy', u'leathery', u'lemon',
u'lily', u'malty', u'meaty', u'medicinal', u'melon', u'metallic', u'milky',
u'mint', u'muguet', u'mushroom', u'musk', u'musty', u'natural', u'nutty',
u'odorless', u'oily', u'onion', u'orange', u'orangeflower', u'orris',
u'ozone', u'peach', u'pear', u'phenolic', u'pine', u'pineapple', u'plum',
u'popcorn', u'potato', u'powdery', u'pungent', u'radish', u'raspberry',
u'ripe', u'roasted', u'rose', u'rummy', u'sandalwood', u'savory', u'sharp',
u'smoky', u'soapy', u'solvent', u'sour', u'spicy', u'strawberry',
u'sulfurous', u'sweaty', u'sweet', u'tea', u'terpenic', u'tobacco',
u'tomato', u'tropical', u'vanilla', u'vegetable', u'vetiver', u'violet',
u'warm', u'waxy', u'weedy', u'winey', u'woody'
]

MONELL_CLASS_LIST = [
    'Green', 'Grassy', 'Cucumber', 'Tomato', 'Hay', 'Herbal', 'Mint',
    'Woody', 'Pine', 'Floral', 'Jasmine', 'Rose', 'Honey', 'Fruity',
    'Citrus', 'Lemon', 'Orange', 'Tropical', 'Berry', 'Peach', 'Apple',
    'Sour', 'Fermented', 'Alcoholic', 'Winey', 'Rummy', 'Caramellic',
    'Vanilla', 'Spicy', 'Coffee', 'Smoky', 'Roasted', 'Meaty', 'Nutty',
    'Fatty', 'Coconut', 'Waxy', 'Dairy', 'Buttery', 'Cheesy', 'Sulfurous',
    'Garlic', 'Earthy', 'Musty', 'Animal', 'Musk', 'Powdery', 'Sweet',
    'Cooling', 'Sharp', 'Medicinal', 'Camphoreous', 'Metallic', 'Ozone',
    'Fishy'
]

CLUSTERED_MONELL_CLASS_LIST = ['Fishy',
'Sour', 'Fermented',
'Earthy', 'Musty', 'Metallic', 'Ozone', 'Smoky', 'Animal',
'Mint', 'Cooling', 'Medicinal', 'Camphoreous', 'Alcoholic', 'Sharp',
'Cucumber', 'Spicy', 'Green', 'Grassy', 'Pine', 'Herbal', 'Hay', 'Woody',
'Coconut', 'Coffee', 'Caramellic', 'Vanilla',
'Buttery', 'Dairy', 'Cheesy', 'Tomato', 'Garlic',
'Fatty', 'Meaty', 'Sulfurous', 'Roasted', 'Nutty',
'Lemon', 'Citrus', 'Orange',
'Sweet', 'Peach', 'Tropical', 'Apple', 'Fruity', 'Berry', 'Honey', 'Winey', 'Rummy',
'Floral', 'Jasmine', 'Rose', 'Musk', 'Powdery', 'Waxy',
]

import collections
assert set(MONELL_CLASS_LIST) == set(CLUSTERED_MONELL_CLASS_LIST), "Missing labels {}".format(set(MONELL_CLASS_LIST) ^ CLUSTERED_MONELL_CLASS_LIST)
assert len(MONELL_CLASS_LIST) == len(CLUSTERED_MONELL_CLASS_LIST), "Duplicate labels {} {}".format(collections.Counter(MONELL_CLASS_LIST).most_common(3),
                                                                                                   collections.Counter(CLUSTERED_MONELL_CLASS_LIST).most_common(3))

def translate_monell_google_odor(odor_name):
    return {'Jasmine': 'jasmin'}.get(odor_name, odor_name).lower()

CLASS_LIST = [translate_monell_google_odor(c) for c in MONELL_CLASS_LIST]

def set_visual_settings(dpi: int = 75, usetex: bool = False, font: str = 'Fira Sans') -> None:
    """"Change matplotlib settings."""
    sns.set(font_scale=1.25)
    sns.set_style('whitegrid')
    IPython.display.set_matplotlib_formats('retina')
    pd.set_option('display.precision', 3)

def print_module_versions(module_list):
    """Print module versions"""
    for module in module_list:
        print(f'{module.__name__:<10s}: {module.__version__}')
