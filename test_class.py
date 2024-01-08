#!/usr/bin/env python3

from traceback import print_exc
import warnings

warnings.filterwarnings("ignore", category=Warning)


import sys
import pytest_dependency
import pytest
from pathlib import Path
from pprint import pp
import numpy as np
import re

#
#
#
#
#

sys.argv = sys.argv[:1]
import src.train as yours

sys.argv += "--modelname cvae_transformer_rc_rcxyz_kl --pose_rep rot6d --lambda_kl 1e-5 --jointstype vertices --batch_size 20 --num_frames 60 --num_layers 8 --lr 0.0001 --glob --translation --no-vertstrans --dataset posefall --num_epochs 1000 --snapshot 100 --folder exps/test".split(
    " "
)
import ACTOR.src.train.train_cvae as actor


from typing import Any
from torch import Tensor

#
#
#
#
#


dependency = pytest.mark.dependency


yours_model = None
yours_optimizer = None
yours_loaded_weights_ = None
yours_loaded_weights = None
actor_model = None
actor_datasets = None
actor_parameters = None
actor_optimizer = None
actor_writer = None
actor_loaded_weights_ = None
actor_loaded_weights = None


@dependency()
def test_init():
    global yours_model, yours_optimizer, yours_loaded_weights_, yours_loaded_weights, actor_model, actor_datasets, actor_parameters, actor_optimizer, actor_writer, actor_loaded_weights_, actor_loaded_weights

    d1 = d2 = None
    yours_loaded_weights_ = (
        yours_loaded_weights
    ) = actor_loaded_weights_ = actor_loaded_weights = {}

    try:
        (
            yours_model,
            yours_optimizer,
            yours_loaded_weights_,
            yours_loaded_weights,
        ) = yours.init()
        # yours_model = yours_model
        # yours_optimizer = yours_optimizer
        # yours_loaded_weights_ = yours_loaded_weights_
        # yours_loaded_weights = yours_loaded_weights
    except InterruptedError as e:
        print_exc()
        d1 = e.args[0]

    try:
        (
            actor_model,
            actor_datasets,
            actor_parameters,
            actor_optimizer,
            actor_writer,
            actor_loaded_weights_,
            actor_loaded_weights,
        ) = actor.init()
        # actor_model = actor_model
        # actor_datasets = actor_datasets
        # actor_parameters = actor_parameters
        # actor_optimizer = actor_optimizer
        # actor_writer = actor_writer
        # actor_loaded_weights_ = actor_loaded_weights_
        # actor_loaded_weights = actor_loaded_weights
    except InterruptedError as e:
        print_exc()
        d2 = e.args[0]

    compare_dict_of_tensors(yours_loaded_weights_, actor_loaded_weights_)
    compare_dict_of_tensors(yours_loaded_weights, actor_loaded_weights)

    compare_dict_of_tensors(
        yours_model.state_dict(),
        actor_model.state_dict(),
        allow_extra_1=True,
        ignore_regex=r".*action(_b|B)iases",
    )

    if d1 is not None and d2 is not None:
        cmp_d(d1, d2)


@dependency(depends=[test_init.__name__])
def test_run_epoch():
    try:
        yours.main(yours_model, yours_optimizer)
    except InterruptedError as e:
        print_exc()
        d1 = e.args[0]

    try:
        actor.my_do_epochs(
            actor_model,
            actor_datasets,
            actor_parameters,
            actor_optimizer,
            actor_writer,
        )
    except InterruptedError as e:
        print_exc()
        d2 = e.args[0]

    if d1 is not None and d2 is not None:
        cmp_d(d1, d2)


#
#
#
#
#


def cmp_d(l1: "list[tuple[str, Any]]", l2: "list[tuple[str, Any]]") -> None:
    for i in range(len(l1)):
        k, v1 = l1[i]
        _, v2 = l2[i]
        print(f"\t>>>\tTesting: {k}")

        assert l1[i][0] == l2[i][0], f"different key! {l1[i][0]} vs {l2[i][0]}"
        assert type(v1) == type(v2), f"different type! {type(v1)} vs {type(v2)}"
        if isinstance(v1, dict):
            return cmp_d(v1, v2)
        elif isinstance(v1, Tensor):
            assert v1.equal(v2), f"{k} is not equal as Tensor"
        elif isinstance(v1, np.ndarray):
            assert np.allclose(v1, v2), f"{k} is not equal as ndarray"
        else:
            assert v1 == v2, f"{k} is not equal!"
        print(f"\t>>>\tPASS: {k}")


def compare_dict_of_tensors(
    dict1: "dict[Any, Tensor]",
    dict2: "dict[Any, Tensor]",
    allow_extra_1: bool = False,
    allow_extra_2: bool = False,
    ignore_regex: str = "",
):
    d1k = set(
        k for k in dict1.keys() if not ignore_regex or not re.match(ignore_regex, k)
    )
    d2k = set(
        k for k in dict2.keys() if not ignore_regex or not re.match(ignore_regex, k)
    )

    if allow_extra_1:
        diff = d1k - d2k
        if diff:
            print(f"dict1 has keys that dict2 does not have: ")
            pp(diff)
    else:
        assert not (d1k - d2k), f"dict1 has keys that dict2 does not have: {d1k - d2k}"

    if allow_extra_2:
        diff = d2k - d1k
        if diff:
            print(f"dict2 has keys that dict1 does not have: ")
            pp(diff)
    else:
        assert not (d2k - d1k), f"dict2 has keys that dict1 does not have: {d2k - d1k}"

    for k in d1k:
        if k in d2k:
            assert dict1[k].allclose(dict2[k]), f"{k} is not equal"
