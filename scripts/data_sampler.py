from torch.utils.data import Dataset, Sampler
from collections import defaultdict
import random
import pandas as pd

class UniqueReportBatchSampler(Sampler):
    """
    Ensures each batch contains at most one image sequence per report.

    Use with:
        DataLoader(dataset, batch_sampler=sampler, ...)

    Do not also pass batch_size=... or shuffle=... to DataLoader.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        report_id_column="Report ID",
        fallback_to_text=True,
        drop_last=False,
        shuffle=True,
        seed=None,
    ):
        if not hasattr(dataset, "df"):
            raise TypeError("UniqueReportBatchSampler expects dataset.df to exist.")

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.report_id_column = report_id_column
        self.fallback_to_text = fallback_to_text
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        df = dataset.df
        groups = defaultdict(list)

        for idx, row in df.iterrows():
            if report_id_column in df.columns and pd.notna(row[report_id_column]):
                report_key = row[report_id_column]
            elif fallback_to_text:
                report_key = row[dataset.text_column]
            else:
                raise ValueError(f"Missing report ID for row {idx}")

            groups[str(report_key).strip()].append(int(idx))

        self.groups = dict(groups)

        if self.drop_last and len(self.groups) < self.batch_size:
            raise ValueError(
                f"Cannot make a full batch of size {self.batch_size} with unique reports. "
                f"Only found {len(self.groups)} unique report groups."
            )

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        if self.seed is None:
            rng = random.Random()
        else:
            rng = random.Random(int(self.seed) + int(self.epoch))

        remaining = {k: v.copy() for k, v in self.groups.items()}

        if self.shuffle:
            for indices in remaining.values():
                rng.shuffle(indices)

        while remaining:
            keys = list(remaining.keys())

            if self.shuffle:
                rng.shuffle(keys)

            if self.drop_last and len(keys) < self.batch_size:
                break

            batch = []

            for key in keys:
                batch.append(remaining[key].pop())

                if len(remaining[key]) == 0:
                    del remaining[key]

                if len(batch) == self.batch_size:
                    break

            if len(batch) == self.batch_size or not self.drop_last:
                yield batch

    def __len__(self):
        counts = [len(v) for v in self.groups.values()]
        n_batches = 0

        while counts:
            active = len(counts)
            take = min(self.batch_size, active)

            if self.drop_last and take < self.batch_size:
                break

            n_batches += 1

            counts.sort(reverse=True)
            counts = [c - 1 for c in counts[:take]] + counts[take:]
            counts = [c for c in counts if c > 0]

        return n_batches