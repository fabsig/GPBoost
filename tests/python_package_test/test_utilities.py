# coding: utf-8
import logging

import numpy as np
import gpboost as gpb


def test_register_logger(tmp_path):
    logger = logging.getLogger("GPBoost")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s | %(message)s')
    log_filename = str(tmp_path / "GPBoost_test_logger.log")
    file_handler = logging.FileHandler(log_filename, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    def dummy_metric(_, __):
        logger.debug('In dummy_metric')
        return 'dummy_metric', 1, True

    gpb.register_logger(logger)

    X = np.array([[1, 2, 3],
                  [1, 2, 4],
                  [1, 2, 4],
                  [1, 2, 3]],
                 dtype=np.float32)
    y = np.array([0, 1, 1, 0])
    gpb_data = gpb.Dataset(X, y)

    eval_records = {}
    gpb.train({'objective': 'binary', 'metric': ['auc', 'binary_error']},
              gpb_data, num_boost_round=10, feval=dummy_metric,
              valid_sets=[gpb_data], evals_result=eval_records,
              categorical_feature=[1], early_stopping_rounds=4, verbose_eval=2)

    gpb.plot_metric(eval_records)

    expected_log = r"""
WARNING | categorical_feature in Dataset is overridden.
New categorical_feature is [1]
INFO | [GPBoost] [Warning] There are no meaningful features, as all feature values are constant.
INFO | [GPBoost] [Info] Number of positive: 2, number of negative: 2
INFO | [GPBoost] [Info] Total Bins 0
INFO | [GPBoost] [Info] Number of data points in the train set: 4, number of used features: 0
INFO | [GPBoost] [Info] [binary:BoostFromScore]: pavg=0.500000 -> initscore=0.000000
INFO | [GPBoost] [Warning] Stopped training because there are no more leaves that meet the split requirements
DEBUG | In dummy_metric
INFO | Training until validation scores don't improve for 4 rounds
INFO | [GPBoost] [Warning] Stopped training because there are no more leaves that meet the split requirements
DEBUG | In dummy_metric
INFO | [2]	training's auc: 0.5	training's binary_error: 0.5	training's dummy_metric: 1
INFO | [GPBoost] [Warning] Stopped training because there are no more leaves that meet the split requirements
DEBUG | In dummy_metric
INFO | [GPBoost] [Warning] Stopped training because there are no more leaves that meet the split requirements
DEBUG | In dummy_metric
INFO | [4]	training's auc: 0.5	training's binary_error: 0.5	training's dummy_metric: 1
INFO | [GPBoost] [Warning] Stopped training because there are no more leaves that meet the split requirements
DEBUG | In dummy_metric
INFO | [GPBoost] [Warning] Stopped training because there are no more leaves that meet the split requirements
DEBUG | In dummy_metric
INFO | [6]	training's auc: 0.5	training's binary_error: 0.5	training's dummy_metric: 1
INFO | [GPBoost] [Warning] Stopped training because there are no more leaves that meet the split requirements
DEBUG | In dummy_metric
INFO | [GPBoost] [Warning] Stopped training because there are no more leaves that meet the split requirements
DEBUG | In dummy_metric
INFO | [8]	training's auc: 0.5	training's binary_error: 0.5	training's dummy_metric: 1
INFO | [GPBoost] [Warning] Stopped training because there are no more leaves that meet the split requirements
DEBUG | In dummy_metric
INFO | [GPBoost] [Warning] Stopped training because there are no more leaves that meet the split requirements
DEBUG | In dummy_metric
INFO | [10]	training's auc: 0.5	training's binary_error: 0.5	training's dummy_metric: 1
INFO | Did not meet early stopping. Best iteration is:
[1]	training's auc: 0.5	training's binary_error: 0.5	training's dummy_metric: 1
WARNING | More than one metric available, picking one to plot.
""".strip()

    gpu_lines = [
        "INFO | [GPBoost] [Info] This is the GPU trainer",
        "INFO | [GPBoost] [Info] Using GPU Device:",
        "INFO | [GPBoost] [Info] Compiling OpenCL Kernel with 16 bins...",
        "INFO | [GPBoost] [Info] GPU programs have been built",
        "INFO | [GPBoost] [Warning] GPU acceleration is disabled because no non-trivial dense features can be found",
        "INFO | [GPBoost] [Warning] Using sparse features with CUDA is currently not supported.",
        "INFO | [GPBoost] [Warning] CUDA currently requires double precision calculations.",
        "INFO | [GPBoost] [Info] GPBoost using CUDA trainer with DP float!!"
    ]
    with open(log_filename, "rt", encoding="utf-8") as f:
        actual_log = f.read().strip()
        actual_log_wo_gpu_stuff = []
        for line in actual_log.split("\n"):
            if not any(line.startswith(gpu_line) for gpu_line in gpu_lines):
                actual_log_wo_gpu_stuff.append(line)

    assert "\n".join(actual_log_wo_gpu_stuff) == expected_log
