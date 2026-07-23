from __future__ import annotations

import numpy as np
import pytest

from tools.local_labels.cutlass_successor_reference import reference_matrix


def test_reference_is_deterministic_and_scoped():
    first = reference_matrix(
        dtype_contract="fp32_inputs_fp32_accumulator",
        batch_count=4,
        m=32,
        n=32,
        k=32,
        seed=42,
    )
    second = reference_matrix(
        dtype_contract="fp32_inputs_fp32_accumulator",
        batch_count=4,
        m=32,
        n=32,
        k=32,
        seed=42,
    )
    assert first.shape == (4, 32, 32)
    assert np.array_equal(first, second)


def test_reference_rejects_unknown_dtype():
    with pytest.raises(ValueError):
        reference_matrix(
            dtype_contract="unknown",  # type: ignore[arg-type]
            batch_count=1,
            m=2,
            n=2,
            k=2,
            seed=42,
        )
