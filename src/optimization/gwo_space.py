#!/usr/bin/env python3

import argparse
import json
import math
import random
from pathlib import Path


EXPECTED_NAMES = [
    "hidden_size",
    "num_layers",
    "dropout",
    "learning_rate",
    "weight_decay",
]


def load_spec(path):
    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def clip01(value):
    return min(
        1.0,
        max(
            0.0,
            float(value),
        ),
    )


def decode_categorical(value, values):
    u = clip01(value)

    index = min(
        int(
            math.floor(
                u * len(values)
            )
        ),
        len(values) - 1,
    )

    return values[index]


def decode_log_uniform(
    value,
    minimum,
    maximum,
):
    u = clip01(value)

    log_min = math.log10(
        float(minimum)
    )

    log_max = math.log10(
        float(maximum)
    )

    return 10.0 ** (
        log_min
        + u * (
            log_max - log_min
        )
    )


def dimension_map(spec):
    dimensions = spec["dimensions"]

    names = [
        item["name"]
        for item in dimensions
    ]

    if names != EXPECTED_NAMES:
        raise ValueError(
            "Unexpected dimension order: "
            + repr(names)
        )

    return {
        item["name"]: item
        for item in dimensions
    }


def decode_position(
    position,
    spec,
):
    if len(position) != 5:
        raise ValueError(
            "Expected a 5-dimensional position."
        )

    dimensions = dimension_map(
        spec
    )

    hidden_size = int(
        decode_categorical(
            position[0],
            dimensions[
                "hidden_size"
            ]["values"],
        )
    )

    num_layers = int(
        decode_categorical(
            position[1],
            dimensions[
                "num_layers"
            ]["values"],
        )
    )

    dropout = float(
        decode_categorical(
            position[2],
            dimensions[
                "dropout"
            ]["values"],
        )
    )

    if num_layers == 1:
        dropout = 0.0

    learning_rate = float(
        decode_log_uniform(
            position[3],
            dimensions[
                "learning_rate"
            ]["minimum"],
            dimensions[
                "learning_rate"
            ]["maximum"],
        )
    )

    weight_decay = float(
        decode_log_uniform(
            position[4],
            dimensions[
                "weight_decay"
            ]["minimum"],
            dimensions[
                "weight_decay"
            ]["maximum"],
        )
    )

    candidate = {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
    }

    validate_candidate(
        candidate,
        spec,
    )

    return candidate


def validate_candidate(
    candidate,
    spec,
):
    dimensions = dimension_map(
        spec
    )

    if candidate["hidden_size"] not in (
        dimensions[
            "hidden_size"
        ]["values"]
    ):
        raise ValueError(
            "Invalid hidden_size."
        )

    if candidate["num_layers"] not in (
        dimensions[
            "num_layers"
        ]["values"]
    ):
        raise ValueError(
            "Invalid num_layers."
        )

    if candidate["dropout"] not in (
        dimensions[
            "dropout"
        ]["values"]
    ):
        raise ValueError(
            "Invalid dropout."
        )

    if (
        candidate["num_layers"] == 1
        and candidate["dropout"] != 0.0
    ):
        raise ValueError(
            "Single-layer candidate must "
            "have dropout=0."
        )

    for name in [
        "learning_rate",
        "weight_decay",
    ]:
        value = float(
            candidate[name]
        )

        minimum = float(
            dimensions[name][
                "minimum"
            ]
        )

        maximum = float(
            dimensions[name][
                "maximum"
            ]
        )

        tolerance = (
            1e-12
            * max(
                1.0,
                abs(maximum),
            )
        )

        if not (
            minimum - tolerance
            <= value
            <= maximum + tolerance
        ):
            raise ValueError(
                f"{name} outside bounds."
            )


def candidate_key(candidate):
    canonical = {
        "dropout": float(
            candidate["dropout"]
        ),
        "hidden_size": int(
            candidate["hidden_size"]
        ),
        "learning_rate": float(
            candidate[
                "learning_rate"
            ]
        ),
        "num_layers": int(
            candidate["num_layers"]
        ),
        "weight_decay": float(
            candidate["weight_decay"]
        ),
    }

    return json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
    )


def gru_parameter_count(
    hidden_size,
    num_layers,
    input_size=8,
    output_size=5,
):
    h = int(hidden_size)
    layers = int(num_layers)

    first_layer = (
        3 * h * input_size
        + 3 * h * h
        + 6 * h
    )

    subsequent_layer = (
        6 * h * h
        + 6 * h
    )

    recurrent = (
        first_layer
        + max(
            0,
            layers - 1,
        )
        * subsequent_layer
    )

    output = (
        output_size * h
        + output_size
    )

    return recurrent + output


def trainer_arguments(
    candidate,
    spec,
):
    fixed = spec[
        "fixed_training"
    ]

    return [
        "--seed",
        str(
            fixed["trainer_seed"]
        ),
        "--hidden-size",
        str(
            candidate["hidden_size"]
        ),
        "--num-layers",
        str(
            candidate["num_layers"]
        ),
        "--dropout",
        repr(
            candidate["dropout"]
        ),
        "--learning-rate",
        repr(
            candidate[
                "learning_rate"
            ]
        ),
        "--weight-decay",
        repr(
            candidate[
                "weight_decay"
            ]
        ),
        "--max-epochs",
        str(
            fixed["max_epochs"]
        ),
        "--patience",
        str(
            fixed["patience"]
        ),
        "--min-delta",
        repr(
            fixed["min_delta"]
        ),
    ]


def categorical_midpoint(
    value,
    values,
):
    index = values.index(
        value
    )

    return (
        index + 0.5
    ) / len(values)


def inverse_log_uniform(
    value,
    minimum,
    maximum,
):
    return (
        math.log10(value)
        - math.log10(minimum)
    ) / (
        math.log10(maximum)
        - math.log10(minimum)
    )


def baseline_reference_position(
    spec,
):
    dimensions = dimension_map(
        spec
    )

    baseline = spec[
        "baseline_reference"
    ]

    return [
        categorical_midpoint(
            baseline[
                "hidden_size"
            ],
            dimensions[
                "hidden_size"
            ]["values"],
        ),
        categorical_midpoint(
            baseline[
                "num_layers"
            ],
            dimensions[
                "num_layers"
            ]["values"],
        ),
        categorical_midpoint(
            baseline[
                "dropout"
            ],
            dimensions[
                "dropout"
            ]["values"],
        ),
        inverse_log_uniform(
            baseline[
                "learning_rate"
            ],
            dimensions[
                "learning_rate"
            ]["minimum"],
            dimensions[
                "learning_rate"
            ]["maximum"],
        ),
        inverse_log_uniform(
            baseline[
                "weight_decay"
            ],
            dimensions[
                "weight_decay"
            ]["minimum"],
            dimensions[
                "weight_decay"
            ]["maximum"],
        ),
    ]


def self_test(spec):
    checks = []

    def check(
        condition,
        label,
    ):
        print(
            label + ":",
            "PASS"
            if condition
            else "FAIL",
        )

        checks.append(
            bool(condition)
        )

    check(
        spec["position_dimension"]
        == 5,
        "POSITION_DIMENSION_5",
    )

    dimensions = dimension_map(
        spec
    )

    check(
        dimensions[
            "hidden_size"
        ]["values"]
        == [16, 32, 64, 96],
        "HIDDEN_CHOICES",
    )

    check(
        dimensions[
            "num_layers"
        ]["values"]
        == [1, 2, 3],
        "LAYER_CHOICES",
    )

    check(
        dimensions[
            "dropout"
        ]["values"]
        == [
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
        ],
        "DROPOUT_CHOICES",
    )

    baseline_position = (
        baseline_reference_position(
            spec
        )
    )

    baseline_candidate = (
        decode_position(
            baseline_position,
            spec,
        )
    )

    print(
        "baseline_position:",
        baseline_position,
    )

    print(
        "baseline_decoded:",
        baseline_candidate,
    )

    reference = spec[
        "baseline_reference"
    ]

    check(
        baseline_candidate[
            "hidden_size"
        ]
        == reference[
            "hidden_size"
        ],
        "BASELINE_HIDDEN_EXACT",
    )

    check(
        baseline_candidate[
            "num_layers"
        ]
        == reference[
            "num_layers"
        ],
        "BASELINE_LAYERS_EXACT",
    )

    check(
        baseline_candidate[
            "dropout"
        ]
        == reference[
            "dropout"
        ],
        "BASELINE_DROPOUT_EXACT",
    )

    check(
        math.isclose(
            baseline_candidate[
                "learning_rate"
            ],
            reference[
                "learning_rate"
            ],
            rel_tol=1e-14,
            abs_tol=0.0,
        ),
        "BASELINE_LR_EXACT",
    )

    check(
        math.isclose(
            baseline_candidate[
                "weight_decay"
            ],
            reference[
                "weight_decay"
            ],
            rel_tol=1e-14,
            abs_tol=0.0,
        ),
        "BASELINE_WD_EXACT",
    )

    baseline_params = (
        gru_parameter_count(
            baseline_candidate[
                "hidden_size"
            ],
            baseline_candidate[
                "num_layers"
            ],
        )
    )

    print(
        "baseline_parameter_count:",
        baseline_params,
    )

    check(
        baseline_params == 4197,
        "BASELINE_PARAMETER_COUNT_4197",
    )

    check(
        gru_parameter_count(
            32,
            2,
        ) == 10533,
        "TWO_LAYER_32_PARAMETER_COUNT_10533",
    )

    maximum_params = max(
        gru_parameter_count(
            hidden,
            layers,
        )
        for hidden in dimensions[
            "hidden_size"
        ]["values"]
        for layers in dimensions[
            "num_layers"
        ]["values"]
    )

    print(
        "maximum_parameter_count:",
        maximum_params,
    )

    check(
        maximum_params == 142757,
        "MAX_PARAMETER_COUNT_142757",
    )

    # Single-layer dropout must collapse to zero.
    one_layer_high_dropout = (
        decode_position(
            [
                0.3,
                0.0,
                1.0,
                0.5,
                0.5,
            ],
            spec,
        )
    )

    check(
        one_layer_high_dropout[
            "dropout"
        ] == 0.0,
        "SINGLE_LAYER_DROPOUT_CANONICALIZATION",
    )

    # Bounds.
    lower = decode_position(
        [0, 0, 0, 0, 0],
        spec,
    )

    upper = decode_position(
        [1, 1, 1, 1, 1],
        spec,
    )

    print(
        "lower_boundary:",
        lower,
    )

    print(
        "upper_boundary:",
        upper,
    )

    check(
        lower[
            "learning_rate"
        ]
        == 0.0001,
        "LR_LOWER_BOUND",
    )

    check(
        math.isclose(
            upper[
                "learning_rate"
            ],
            0.005,
            rel_tol=1e-14,
        ),
        "LR_UPPER_BOUND",
    )

    check(
        lower[
            "weight_decay"
        ]
        == 0.000001,
        "WD_LOWER_BOUND",
    )

    check(
        math.isclose(
            upper[
                "weight_decay"
            ],
            0.001,
            rel_tol=1e-14,
        ),
        "WD_UPPER_BOUND",
    )

    # Random decoder stress test.
    rng = random.Random(
        20260811
    )

    unique_keys = set()

    for _ in range(1000):
        position = [
            rng.random()
            for _ in range(5)
        ]

        candidate = (
            decode_position(
                position,
                spec,
            )
        )

        validate_candidate(
            candidate,
            spec,
        )

        unique_keys.add(
            candidate_key(
                candidate
            )
        )

    print(
        "stress_unique_candidate_keys:",
        len(unique_keys),
    )

    check(
        len(unique_keys) > 100,
        "DECODER_STRESS_TEST",
    )

    # Budget arithmetic.
    budget = spec[
        "search_budget"
    ]

    methods = budget[
        "methods"
    ]

    expected_fold_trainings = (
        len(methods)
        * len(
            budget[
                "optimizer_seeds"
            ]
        )
        * budget[
            "n_outer_contexts"
        ]
        * budget[
            "fitness_evaluation_budget_per_search"
        ]
        * budget[
            "n_inner_folds_per_candidate"
        ]
    )

    print(
        "maximum_theoretical_fold_trainings:",
        expected_fold_trainings,
    )

    check(
        expected_fold_trainings
        == 24750,
        "BUDGET_ARITHMETIC_24750",
    )

    check(
        budget[
            "standard_or_no_m1_accounting"
        ][
            "total_evaluations"
        ] == 30,
        "STANDARD_BUDGET_30",
    )

    check(
        budget[
            "m1_accounting"
        ][
            "total_evaluations"
        ] == 30,
        "M1_BUDGET_30",
    )

    check(
        spec[
            "outer_test_policy"
        ][
            "available_to_optimizer"
        ] is False,
        "OUTER_TEST_NOT_AVAILABLE",
    )

    check(
        spec[
            "outer_test_policy"
        ][
            "used_for_fitness"
        ] is False,
        "OUTER_TEST_NOT_IN_FITNESS",
    )

    if not all(checks):
        raise SystemExit(1)

    print(
        "GWO_SPACE_SELF_TEST: PASS"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--spec",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
    )

    parser.add_argument(
        "--position",
        nargs=5,
        type=float,
    )

    args = parser.parse_args()

    spec = load_spec(
        args.spec
    )

    if args.self_test:
        self_test(spec)

    if args.position is not None:
        candidate = decode_position(
            args.position,
            spec,
        )

        print(
            json.dumps(
                {
                    "candidate": candidate,
                    "candidate_key": candidate_key(
                        candidate
                    ),
                    "parameter_count": (
                        gru_parameter_count(
                            candidate[
                                "hidden_size"
                            ],
                            candidate[
                                "num_layers"
                            ],
                        )
                    ),
                    "trainer_arguments": (
                        trainer_arguments(
                            candidate,
                            spec,
                        )
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
