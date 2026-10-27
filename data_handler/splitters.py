import random


def split_subjects(subject_ids, seed, val_split, test_split):
    if val_split + test_split >= 1.0:
        raise ValueError(
            f"val_split ({val_split}) + test_split ({test_split}) must sum to < 1.0"
        )

    subject_ids = list(subject_ids)
    if len(subject_ids) < 3:
        raise ValueError(
            f"Need at least 3 subjects for train/val/test split; found {len(subject_ids)}."
        )

    shuffled = subject_ids[:]
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_split))
    n_test = max(1, round(len(shuffled) * test_split))

    while n_val + n_test >= len(shuffled):
        if n_val >= n_test:
            n_val -= 1
        else:
            n_test -= 1
        if n_val < 1 or n_test < 1:
            raise ValueError(
                f"Not enough subjects ({len(shuffled)}) for non-empty train/val/test splits "
                f"with val_split={val_split}, test_split={test_split}."
            )

    return (
        set(shuffled[n_test + n_val:]),
        set(shuffled[n_test:n_test + n_val]),
        set(shuffled[:n_test]),
    )


def records_for_subjects(records, subject_groups):
    return tuple(
        [record for record in records if record["id"] in subjects]
        for subjects in subject_groups
    )
