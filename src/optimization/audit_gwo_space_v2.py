#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def check(condition, label):
    print(
        label + ":",
        "PASS" if condition else "FAIL",
    )

    if not condition:
        raise RuntimeError(label)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--v1",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--v2",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    optimization_dir = Path(
        __file__
    ).resolve().parent

    sys.path.insert(
        0,
        str(optimization_dir),
    )

    from gwo_space import (
        baseline_reference_position,
        candidate_key,
        decode_position,
        dimension_map,
        gru_parameter_count,
        load_spec,
        validate_candidate,
    )

    v1 = load_spec(
        args.v1
    )

    v2 = load_spec(
        args.v2
    )

    d1 = dimension_map(
        v1
    )

    d2 = dimension_map(
        v2
    )

    old_hidden = d1[
        "hidden_size"
    ]["values"]

    new_hidden = d2[
        "hidden_size"
    ]["values"]

    print(
        "v1_hidden_choices:",
        old_hidden,
    )

    print(
        "v2_hidden_choices:",
        new_hidden,
    )

    check(
        old_hidden
        == [
            16,
            32,
            64,
            96,
        ],
        "V1_HIDDEN_SPACE_PRESERVED",
    )

    check(
        new_hidden
        == [
            16,
            32,
            64,
            96,
            128,
            160,
            192,
            224,
            256,
        ],
        "V2_HIDDEN_SPACE_EXPANDED",
    )

    check(
        d1["num_layers"]["values"]
        == d2["num_layers"]["values"]
        == [1, 2, 3],
        "LAYERS_UNCHANGED",
    )

    check(
        d1["dropout"]["values"]
        == d2["dropout"]["values"]
        == [
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
        ],
        "DROPOUT_UNCHANGED",
    )

    check(
        d1[
            "learning_rate"
        ]
        == d2[
            "learning_rate"
        ],
        "LEARNING_RATE_SPACE_UNCHANGED",
    )

    check(
        d1[
            "weight_decay"
        ]
        == d2[
            "weight_decay"
        ],
        "WEIGHT_DECAY_SPACE_UNCHANGED",
    )

    baseline_position = (
        baseline_reference_position(
            v2
        )
    )

    baseline = decode_position(
        baseline_position,
        v2,
    )

    print(
        "v2_baseline_position:",
        baseline_position,
    )

    print(
        "v2_baseline_decoded:",
        baseline,
    )

    check(
        baseline[
            "hidden_size"
        ] == 32,
        "BASELINE_HIDDEN_32",
    )

    check(
        baseline[
            "num_layers"
        ] == 1,
        "BASELINE_LAYERS_1",
    )

    check(
        baseline[
            "dropout"
        ] == 0.0,
        "BASELINE_DROPOUT_0",
    )

    baseline_params = (
        gru_parameter_count(
            32,
            1,
        )
    )

    print(
        "baseline_parameter_count:",
        baseline_params,
    )

    check(
        baseline_params
        == 4197,
        "BASELINE_PARAMETERS_4197",
    )

    maximum_params = max(
        gru_parameter_count(
            hidden,
            layers,
        )
        for hidden
        in new_hidden
        for layers
        in d2[
            "num_layers"
        ]["values"]
    )

    print(
        "v2_maximum_parameter_count:",
        maximum_params,
    )

    check(
        maximum_params
        == 995077,
        "V2_MAX_PARAMETERS_995077",
    )

    lower = decode_position(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        v2,
    )

    upper = decode_position(
        [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ],
        v2,
    )

    print(
        "v2_lower_boundary:",
        lower,
    )

    print(
        "v2_upper_boundary:",
        upper,
    )

    check(
        lower[
            "hidden_size"
        ] == 16,
        "V2_LOWER_HIDDEN_16",
    )

    check(
        upper[
            "hidden_size"
        ] == 256,
        "V2_UPPER_HIDDEN_256",
    )

    check(
        upper[
            "num_layers"
        ] == 3,
        "V2_UPPER_LAYERS_3",
    )

    check(
        upper[
            "dropout"
        ] == 0.4,
        "V2_UPPER_DROPOUT_04",
    )

    rng = random.Random(
        20260813
    )

    keys = set()

    for _ in range(
        2000
    ):
        position = [
            rng.random()
            for _ in range(5)
        ]

        candidate = (
            decode_position(
                position,
                v2,
            )
        )

        validate_candidate(
            candidate,
            v2,
        )

        keys.add(
            candidate_key(
                candidate
            )
        )

    print(
        "v2_stress_unique_candidates:",
        len(keys),
    )

    check(
        len(keys) > 200,
        "V2_DECODER_STRESS_TEST",
    )

    check(
        v2[
            "outer_test_policy"
        ][
            "available_to_optimizer"
        ]
        is False,
        "OUTER_TEST_NOT_AVAILABLE",
    )

    check(
        v2[
            "outer_test_policy"
        ][
            "used_for_fitness"
        ]
        is False,
        "OUTER_TEST_NOT_IN_FITNESS",
    )

    rationale = v2[
        "hidden_size_boundary_rationale"
    ]

    check(
        rationale[
            "outer_test_used"
        ]
        is False,
        "BOUNDARY_TEST_USED_NO_OUTER_TEST",
    )

    check(
        rationale[
            "best_hidden_counts"
        ]
        == {
            "96": 0,
            "128": 1,
            "160": 2,
            "192": 2,
            "224": 2,
            "256": 4,
        },
        "BOUNDARY_RESULTS_RECORDED",
    )

    print()
    print(
        "GWO_SPACE_V2_AUDIT: PASS"
    )


if __name__ == "__main__":
    main()
