"""HGPT five-point sparse tracking using the shared, in-repository HGPT evaluator."""
from humantracker.eval.backends import hgpt
from humantracker.eval.backends.hgpt import (
    compute_category_summary,
    compute_overall_summary,
    print_category_summary,
    print_overall_summary,
)
from humantracker.eval.native.hgpt.sparse import (
    G1TrackSparseInferFn,
    NUM_OBSERVATIONS,
    SPARSE_KPT_NAMES,
)
from humantracker.eval.paths import required_file

DEFAULT_POLICY = (
    "storage/checkpoints/trackers/hgpt_sparse/"
    "sparse5pt_best_stage2_4B_cumulative9B.onnx"
)

# qpos contributes only current root xy/yaw commands, not reference joint angles.
# The full reference remains available to the evaluator for whole-body metrics.
REF_FIELDS_CONSUMED = ("qpos", "kpt2gv_pose", "kpt_cvel_in_gv")

OPTIONS = tuple(
    (flag, {**options, "default": DEFAULT_POLICY,
            "help": "HGPT five-point sparse ONNX policy (182-D observation)"})
    if flag == "--policy" else (flag, options)
    for flag, options in hgpt.OPTIONS
)


def validate_policy_session(session):
    """Reject dense/three-point/recurrent policies before any worker is started."""
    inputs = session.get_inputs()
    if (len(inputs) != 1 or inputs[0].name != "obs"
            or inputs[0].type != "tensor(float)"
            or len(inputs[0].shape) != 2 or inputs[0].shape[1] != NUM_OBSERVATIONS
            or isinstance(inputs[0].shape[0], int) and inputs[0].shape[0] != 1):
        raise ValueError("HGPT 5-point policy requires float32 obs with shape [batch, 182]")
    outputs = {node.name: node for node in session.get_outputs()}
    action = outputs.get("continuous_actions")
    if (action is None or action.type != "tensor(float)" or len(action.shape) != 2
            or action.shape[1] != 29
            or isinstance(action.shape[0], int) and action.shape[0] != 1):
        raise ValueError("HGPT 5-point policy requires float32 continuous_actions [batch, 29]")


def validate(args):
    if args.privileged:
        raise ValueError("--privileged is not supported by hgpt_sparse; the policy is non-privileged")
    hgpt.validate(args)
    import onnxruntime as ort
    validate_policy_session(ort.InferenceSession(
        required_file(args.policy), providers=["CPUExecutionProvider"]
    ))


def build_context(args, xml_path):
    context = hgpt.build_context(args, xml_path)
    validate_policy_session(context["policy"].onnx_model)
    context["inference_cls"] = G1TrackSparseInferFn
    context["ref_fields_consumed"] = REF_FIELDS_CONSUMED
    return context


def evaluate(context, task):
    result = hgpt.evaluate(context, task)
    result.update(tracker="hgpt_sparse", sparse_mode="5point",
                  sparse_keypoints=list(SPARSE_KPT_NAMES))
    return result
