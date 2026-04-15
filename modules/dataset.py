# Base Class for datasets

import json
import os
import glob
import re
from tqdm import tqdm

from string import ascii_uppercase
from datasets import load_dataset

import random
import numpy as np

class GSM8KDataset:
    def __init__(self, split="train", subset_size=100, num_bins=10, seed=42):
        self.data = load_dataset(
            "lime-nlp/GSM8K_Difficulty",
            "Difficulty Score",
            split=split
        )

        self.indices = self._create_uniform_subset(
            subset_size=subset_size,
            num_bins=num_bins,
            seed=seed
        )

    def _create_uniform_subset(self, subset_size, num_bins, seed):
        random.seed(seed)
        solved = np.array([x["solved_percentage"] for x in self.data])
        bins = np.quantile(solved, np.linspace(0, 1, num_bins + 1))
        bin_ids = np.digitize(solved, bins, right=True) - 1
        bin_ids = np.clip(bin_ids, 0, num_bins - 1)

        bin_to_indices = {i: [] for i in range(num_bins)}
        for idx, b in enumerate(bin_ids):
            bin_to_indices[b].append(idx)

        samples_per_bin = subset_size // num_bins
        selected_indices = []

        for b in range(num_bins):
            candidates = bin_to_indices[b]
            if len(candidates) == 0:
                continue
            
            chosen = random.sample(
                candidates,
                min(samples_per_bin, len(candidates))
            )
            selected_indices.extend(chosen)

        if len(selected_indices) < subset_size:
            remaining = list(set(range(len(self.data))) - set(selected_indices))
            selected_indices.extend(
                random.sample(remaining, subset_size - len(selected_indices))
            )

        random.shuffle(selected_indices)
        return selected_indices[:subset_size]

    def __len__(self):
        return len(self.indices)

    def _get_answer(self, idx):
        real_idx = self.indices[idx]
        answer = self.data[real_idx]["ground_truth"]
        match = re.search(r"#### (.*)", answer)
        return match.group(1).strip() if match else None

    def _get_difficulty(self, idx):
        real_idx = self.indices[idx]
        return 1. - self.data[real_idx]["solved_percentage"] / 100.0

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return self.data[real_idx]["problem"]

class Dataset:
    def __init__(self, name, dataset_path):
        """
        Args:
            name: name of the dataset
            dataset_path: path to the dataset
        """
        self.name = name
        self.dataset_path = dataset_path
        self.data = self.load_data()

    def __len__(self):
        #return len(self.data)
        return 100
    
    def load_data(self):
        """
        Loads the dataset.
        Returns:
            data: the dataset
        """
        with open(os.path.join(self.dataset_path, "data.json"), 'r') as f:
            data = json.load(f)
        return data

    def format_prompt_basic(self, idx):
        sep = "\n"
        row = self.data[idx]
        evidence = row["weak_evidence"][0]
        prompt = f"""{row["context"]} {evidence}{sep}{row["question"]}{sep}Answer choices:{sep}(A) {row["ans0"]}{sep}(B) {row["ans1"]}{sep}(C) {row["ans2"]}"""
        return prompt

    def __getitem__(self, idx):
        return self.format_prompt_basic(idx)