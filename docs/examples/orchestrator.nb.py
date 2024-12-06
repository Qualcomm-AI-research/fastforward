# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,.nb.py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Neural Network Execution Orchestration
#
# <details>
#   <summary>Notes on <code>fastforward._orchestration</code></summary>
#   <div>
#     <p>
#       <code>fastforward._orchestration</code> is considered internal API. As
#       such, the API may change. As a user, we suggest to use a more high level
#       interface to members of <code>fastforward._orchestration</code>. These
#       docs are primarily included for FastForward developers.
#     </p>
#   </div>
# </details>
#
# ## Introduction
#
# For various training methods of neural networks, especially for quantized neural networks,
# algorithms may require to run network in a non-sequantial manner. The
# `ConcurrentExecOrchestrator` is a utility to help implement these types of algorithms.
# To illustrate this, let's consider the following function:


# %%
from typing import Any, Optional

from IPython.display import Markdown, display

from fastforward._orchestration.concurrent_execution import ConcurrentExecOrchestrator


def example_function(name: str) -> None:
    print(f"#1 Start 'example_function' for {name}.")
    print(f"#2 Do something for {name}.")
    print(f"#3 Clean up for {name}.")
    print(f"#4 Finish 'example_function' for {name}.")


# %% [markdown]
# Let's say we have two input to this function: `alice = "Alice"` and `bob = "Bob"` and we call
# the function sequantially, the output will be as expected:

# %%

people = ["Alice", "Bob"]
alice, bob = people

example_function(alice)
example_function(bob)


# %% [markdown]
# Now, let's say we want to run step #1 and #2 for all people before we run any step #3 or #4.
# We can rewrite our function as follows:


# %%
def example_function_multi(names: list[str]) -> None:
    for name in names:
        print(f"#1 Start 'example_function' for {name}.")
        print(f"#2 Do something for {name}.")
    for name in names:
        print(f"#3 Clean up for {name}.")
        print(f"#4 Finish 'example_function' for {name}.")


example_function_multi(people)

# %% [markdown]
# However, this requires us to rewrite our function for a single person to a function that accepts
# multiple persons. In the case of neural networks, this may be very cumbersome. Ideally, we
# would rewrite our function to something like the following (and also remove
# some of the more verbose labeling):


# %%
def example_function_wait(name: str) -> None:
    print(f"#1 {name}.")
    print(f"#2 {name}.")

    # This function does not exist yet.
    wait_for_all_other_people()

    print(f"#3 {name}.")
    print(f"#4 {name}.")


# %% [markdown]
# The `wait_for_other_people()` function does not exist yet, but we can implement it using
# the help of `ConcurrentExecOrchestrator`.
#

# %%

orchestrator = ConcurrentExecOrchestrator(
    target=example_function_wait,
    num_stages=1,
    execution_order=[(0,)],
)


# Here we implement wait_for_other_people that is used in example_function_wait.
def wait_for_all_other_people():
    orchestrator.synchronize()


# Instead of calling our target function (example_function_wait) directly, we call add_batch()
# on the orchestrator. This will register the arguments and call the target function for each
# registered set of of arguments once the orchestrator starts.
for name in people:
    orchestrator.add_batch(name)

orchestrator.start()

# %% [markdown]
# Note that the code above first ran `example_function_wait` up to `wait_for_all_other_people()`
# for `alice`, then for `bob` and then the remainder for `alice` and then for `bob` -- in order. The code
# above requires some 'ceremony' to set up. For this reason, this API is considered internal
# and should be abstracted in user code for more specific use cases.
#
# # ## Terminology
# In our example above, we used a simple function that waited for other
# executions before completing. This is a use-case of the `ConcurrentExecOrchestrator`, but it
# is more versatile. To keep things organized, let's first introduce some termonology. For this
# we use the following example function:
#
# ```python
#
# def example_target(data):
#     print("partition_1")
#     orchestrator.synchronize()
#     print("partition_2")
#     orchestrator.synchronize()
#     print("partition_3")
# ```
#
# A __target function__ is the function that is executed by the orchestrator. The function acts
# as an entry point, and for each invocation a seperate thread is created.
#
# The target function above is separated in three partition. We consider a __partition__
# _any code that runs between two calls to `synchonize()` or between
# `synchronize()` and the start/end of the target function_
#
# As shown in the introduction, the input to each invocation of the target function is registered
# using `ConcurrentExecOrchestrator.add_batch()`. As the name of this method suggests, we refer
# to each separate 'data' registration as a __batch__.
#
# Finally, the orchestrator has a notion of __stages__ and __execution order__. In our examples
# so far we have used just one stage and the implied execution order `[(0,)]`. Stages refer to
# the number of times the target function is executed per batch. Although partitions always run
# sequentially, i.e., the $n+1$th partition only starts once the $n$th partition finishes for all
# executions (batches $\times$ `num_stages`), the order of stages can be manipulated. For example,
# consider our two batches `alice` and `bob`, a target function with two partitions `p1` and `p2` and
# we'll use two stages `s1` and `s2`. The order of execution could then be any of the following (assuming `alice` denotes the first batch and `bob` the second:
#
# * `alice_s1_p1`, `alice_s2_p2`, `bob_s1_p1`, `bob_s2_p2`
# * `alice_s1_p1`, `bob_s1_p2`, `alice_s2_p1`, `bob_s2_p2`
# * `alice_s2_p1`, `alice_s1_p2`, `bob_s2_p1`, `bob_s1_p2`
# * `alice_s2_p1`, `bob_s2_p2`, `alice_s1_p1`, `bob_s1_p2`
#
# Below we show a concrete example using two batches, two partitions, and three stages.


# %% jupyter={"source_hidden": true}
# Helper Functions


def tabulate_exec_order_outputs(labels: list[str], outputs: list[list[tuple[Any, ...]]]) -> None:
    max_length = max([len(str(label)) for output in outputs for label in output])
    raw_markdown_lines = [
        "|" + "".join(f"{label}|" for label in labels),
        "|" + "".join("---|" for _ in outputs),
    ]
    fmt = "__{0}__: part {1}, stage {2}"
    for lines in zip(*outputs):
        data = " | ".join(fmt.format(*line).ljust(max_length) for line in lines)
        raw_markdown_lines.append(f"|{data}|")
    display(Markdown("\n".join(raw_markdown_lines)))


# %%


class TargetFunction:
    def __init__(self, orchestrator: Optional[ConcurrentExecOrchestrator] = None):
        self.orchestrator: Optional[ConcurrentExecOrchestrator] = orchestrator
        self.outputs: list[tuple[str, int, int]] = []

    def __call__(self, name: str) -> None:
        stage = self.orchestrator.stage if self.orchestrator is not None else -1
        self.outputs.append((name, 1, stage))
        if self.orchestrator is not None:
            self.orchestrator.synchronize()
        self.outputs.append((name, 2, stage))


def execution_order_example(
    num_stages: int, execution_order: list[tuple[int, ...]]
) -> list[tuple[str, int, int]]:
    target_function = TargetFunction()
    orchestrator = ConcurrentExecOrchestrator(
        target_function, num_stages=num_stages, execution_order=execution_order
    )
    target_function.orchestrator = orchestrator

    for batch in people:
        orchestrator.add_batch(batch)
    orchestrator.start()
    return target_function.outputs


exec_orders: list[list[tuple[int, ...]]] = [
    [(0,), (1,), (2,)],
    [(0, 1), (2,)],
    [(0, 1, 2)],
    [(2, 0), (1,)],
]
exec_results = [execution_order_example(3, exec_order) for exec_order in exec_orders]

tabulate_exec_order_outputs(list(map(str, exec_orders)), exec_results)
# %% [markdown]
# It is still the case that for
# each execution a seperate thread is created. For example, when using 3 batches and 4 stages, 12
# threads are created.
#
# ## Error Handling
# If an error occurs during one of the executions, all other executions are
# stopped. Once all executions have terminated a RuntimeError is raised on
# the main thread that references the caught execption in the execution.
#
# ## Hooks
# Hooks can be used to change the behaviour of `ConcurrentExecOrchestrator`.
# In particular there are pre and post 'stage' hooks and pre and post 'global
# stage' hooks. A stage hook is called either before (pre) or after (post)
# the execution of a single stage. I.e., this hook is called
# `num_stages * #batches * #partitions` times. In contrast, the 'global stage'
# hooks the global_pre_stage hook is invoked once per stage between two
# synchronize calls. Consider the same target function as above and the following
# orchestrator setup.
#
# Below we show an example execution using hooks. Since hooks have access to
# the orchestrator, they can manipulate the batch data through the `batch_data`
# attribute of the orchestrator. This allows for further communication and/or
# updates between stages.


# %%
# Hook Example

# The following example shows a 3-stage orchestrator and we will be applying
# global and single stage hooks on the last stage


def hook_target(idx):
    print(f"Partition=1 batch={idx} stage={orchestrator.stage=}")
    orchestrator.synchronize(f"{idx=}")
    print(f"Partition=2 batch={idx} stage={orchestrator.stage}")


orchestrator = ConcurrentExecOrchestrator(hook_target, 3, [(0, 1), (2,)])
orchestrator.add_batch(1)
orchestrator.add_batch(2)


def pre_stage_hook(orchestrator: ConcurrentExecOrchestrator, stage_data: Any) -> None:
    print(f"    >> pre_stage_hook stage={orchestrator.stage} batch={orchestrator.batch}")


def post_stage_hook(orchestrator: ConcurrentExecOrchestrator, stage_data: Any) -> None:
    print(f"    >> post_stage_hook stage={orchestrator.stage} batch={orchestrator.batch}")


def global_pre_stage_hook(orchestrator: ConcurrentExecOrchestrator) -> None:
    print(f"  >>>> global_pre_stage_hook stage={orchestrator.stage}")


def global_post_stage_hook(orchestrator: ConcurrentExecOrchestrator) -> None:
    print(f"  >>>> global_post_stage_hook stage={orchestrator.stage}")


hooked_stage = 2
orchestrator.register_pre_stage_hook(hooked_stage, pre_stage_hook)
orchestrator.register_post_stage_hook(hooked_stage, post_stage_hook)
orchestrator.register_global_pre_stage_hook(hooked_stage, global_pre_stage_hook)
orchestrator.register_global_post_stage_hook(hooked_stage, global_post_stage_hook)


orchestrator.start()


# %% [markdown]
# ## Repeated Stages
#
# In some cases, for example when performing an optimization loop, we may want
# to repeat a single stage multiple times before moving to the next stage. This
# can be achieved by passing `repeat_stage=True` to the `synchronize()` method
# on the orchestrator. If the current batch is the last of the stage and
# `repeat_stage` is True, then instead of going to the next stage, another
# execution of all batches in the stage will follow. Please be aware of the following:
#
# * if `repeat_stage=True` is passed for an execution corresponding to a batch that is not the last batch, it will not have an affect.
# * If `repeat_stage=False` is passed to the penultimate (or any non-last batch) and `repeat_stage=True` is passed for the last batch, the stage will be repeated for __all__ batches.
# * The target function is responsible for executing a (sub-)partition multiple times. If `synchronize()` is called with `repeat_stage=True` without taking care of this, the orchestration fails and may even loop forever.
#
# See below for an example that has a loop in partition 2, but only for stage 1.

# %%
# Repeated stage example


def repeated_target(idx):
    print(f"partition 1 stage={orchestrator.stage} batch={orchestrator.batch}")
    orchestrator.synchronize()
    num_steps = 3 if orchestrator.stage == 1 else 1
    for i in range(num_steps):
        print(f"partition 2 stage={orchestrator.stage} batch={orchestrator.batch} iteration={i}")
        repeat_stage = orchestrator.stage == 1 and i < 2
        orchestrator.synchronize(repeat_stage=repeat_stage)
    print(f"partition 3 stage={orchestrator.stage} batch={orchestrator.batch}")


orchestrator = ConcurrentExecOrchestrator(repeated_target, 3, [(0,), (1,), (2,)])
orchestrator.add_batch(1)
orchestrator.add_batch(2)


orchestrator.start()
