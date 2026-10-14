from __future__ import annotations

from collections import Counter


def validate_episode_ids(episode_ids, source):
    identifiers = list(episode_ids)
    if any(not isinstance(value, str) or not value.strip() for value in identifiers):
        raise RuntimeError(f'Empty or invalid episode ID in {source}')
    counts = Counter(identifiers)
    duplicates = sorted(value for value, count in counts.items() if count > 1)
    if duplicates:
        raise RuntimeError(
            f'Duplicate episode IDs in {source}: {len(identifiers)} records, '
            f'{len(counts)} unique IDs; duplicates: {", ".join(duplicates[:8])}. '
            'Use source-qualified trace_id values rather than trace basenames.'
        )
    return identifiers
