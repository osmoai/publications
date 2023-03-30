import numpy as np
import pandas as pd


class CachedValues():
  """Class for mapping keys to pre-computed values.

  Keys are hashable objects in numpy arrays, for example, SMILES strings.
  """

  def __init__(self, keys: np.ndarray, values: np.ndarray):
    if len(set(keys)) != len(keys):
      raise ValueError('All keys should be distinct; duplicates found!')
    if len(keys) != len(values):
      raise ValueError('Keys and values should have the same length,'
                       ' found len(keys) of {} and len(values) of {}'.format(
                           len(keys), len(values)))
    self.keys = keys
    self.values = values
    self.cache = pd.Series(data=np.arange(len(keys)), index=keys)

  @classmethod
  def load_from_npz(cls,
                    filepath,
                    key_name='smiles',
                    value_name='prediction'):
    data = np.load(filepath, allow_pickle=True)
    return cls(data[key_name], data[value_name])

  def __call__(self, key_array: np.ndarray):
    if not isinstance(key_array, np.ndarray):
      raise ValueError(f'key_array={key_array} should be a numpy array!')
    indices = self.cache[key_array].values
    if np.any(np.isnan(indices)):
      raise ValueError(
          f'key_array={key_array[np.isnan(indices)]} not in cache index!')
    return self.values[indices]

  def save_to_npz(self, filepath, key_name='smiles', value_name='prediction'):
    np.savez_compressed(
        filepath, **{key_name: self.keys, value_name: self.values})
