# run_pipeline_keypoint_real.py
#
# Unified real-image pipeline:
#   original image
#     -> mug detection
#     -> fixed 420x420 canonical crop centered on detector bbox
#     -> keypoint heatmap inference
#     -> restore keypoints to original-image coordinates
#     -> PnP / RANSAC
#
# IMPORTANT:
#   This crop rule matches the real fine-tuning data generation:
#     - crop center = detector bbox center
#     - crop size   = fixed 420x420 original-image pixels
#     - no bbox-dependent scaling
#     - no margin-based resizing
#     - at image boundaries, shift the 420x420 window
#       instead of shrinking/padding it
#
# CSV is LOG ONLY. It is not used to connect pipeline stages.
#
# Saved per frame:
#   *_detection.png
#   *_crop.png
#   *_keypoints_crop.png
#   *_keypoints_original.png
#   *_pnp.png

from pathlib import Path
import argparse
import csv
import json
import time

import cv2
import numpy as np
import torch
from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large,
)

from dataset import (
    INPUT_SIZE,
    HEATMAP_SIZE,
    KEYPOINT_NAMES,
)
from model import MugKeypointHeatmapNet


# ============================================================
# Defaults
# ============================================================

DEFAULT_FRAMES_ROOT = Path(
    "../real_dataset/frames"
)

DEFAULT_DETECTOR_PATH = Path(
    "../detection/models/ssdlite320_sim0550_0552_epoch_030.pth"
)

DEFAULT_KEYPOINT_MODEL_PATH = Path(
    # "model/sim/best_model.pth"
    "model/real/best_model.pth"
)

DEFAULT_OUTPUT_DIR = Path(
    # "output/pipeline_keypoint_real_simonly"
    "output/pipeline_keypoint_real"
)

# Original Labelme annotations are used ONLY for optional
# GT visualization. They are not used by detection/crop/PnP.
DEFAULT_LABELME_ROOT = Path(
    "dataset/real_anno"
)

DETECTOR_NUM_CLASSES = 3
MUG_CLASS_ID = 2
DETECTION_SCORE_THRESHOLD = 0.50

# Must match the canonical crop used by synthetic/real
# keypoint data generation.
CANONICAL_CROP_SIZE = 420

# Imported from dataset.py.
KP_INPUT_SIZE = INPUT_SIZE
KP_HEATMAP_SIZE = HEATMAP_SIZE

OBJECT_POINTS = np.asarray(
    [
        [ 0.036509663, -0.000531428, 0.001519144],
        [-0.036405534, -0.000531428, 0.001519144],
        [-0.000754625, -0.035997842, 0.001519144],
        [-0.000754625,  0.036681805, 0.001519144],
        [ 0.036509663, -0.000531000, 0.088068008],
        [-0.036405534, -0.000531428, 0.088068008],
        [-0.000754625, -0.035997842, 0.088068008],
        [-0.000754625,  0.036681805, 0.088068008],
        [ 0.037754208, -0.000024502, 0.078986347],
        [ 0.037754208, -0.000024502, 0.024735928],
        [ 0.071656495,  0.000063794, 0.061814904],
    ],
    dtype=np.float64,
)

CAMERA_MATRIX = np.asarray(
    [
        [1726.973813, 0.0, 949.848710],
        [0.0, 1727.049940, 519.402872],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

DIST_COEFFS = np.asarray(
    [
        0.191961228,
        -2.27399277,
        0.001726611,
        -0.006301238,
        10.8429528,
    ],
    dtype=np.float64,
)

RANSAC_REPROJ_ERROR = 12.0 #8.0 #5.0
RANSAC_ITERATIONS = 1000
FINAL_INLIER_ERROR = 12.0 #8.0 #5.0
MIN_INLIERS = 6
MIN_TOP_INLIERS = 3
MIN_BOTTOM_INLIERS = 2
MAX_USED_RMSE = 5.0
MAX_ALL_RMSE = 20.0
MAX_REFINE_ROUNDS = 3

TOP_INDICES = {4, 5, 6, 7}
BOTTOM_INDICES = {0, 1, 2, 3}

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def synchronize_device():
    """Wait for queued CUDA work so timing is accurate."""
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


# ============================================================
# Helpers
# ============================================================

def collect_images(
    frames_root,
):
    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
    }

    return [
        p
        for p in sorted(
            frames_root.rglob("*")
        )
        if (
            p.is_file()
            and p.suffix.lower()
            in extensions
        )
    ]


def build_labelme_index(
    labelme_root,
):
    index = {}

    if (
        labelme_root is None
        or not labelme_root.exists()
    ):
        return index

    for json_path in sorted(
        labelme_root.glob("*.json")
    ):
        try:
            with open(
                json_path,
                "r",
                encoding="utf-8",
            ) as f:
                data = json.load(f)

            image_path = data.get(
                "imagePath"
            )

            if image_path:
                index[
                    Path(
                        image_path
                    ).name
                ] = json_path

        except Exception as exc:
            print(
                f"[WARN] Could not read "
                f"{json_path}: {exc}"
            )

    return index


def read_labelme_keypoints(
    json_path,
):
    if json_path is None:
        return {}

    with open(
        json_path,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    result = {}

    for shape in data.get(
        "shapes",
        [],
    ):
        label = shape.get(
            "label"
        )

        if (
            shape.get(
                "shape_type"
            ) == "point"
            and label in KEYPOINT_NAMES
        ):
            points = shape.get(
                "points",
                []
            )

            if not points:
                continue

            x, y = points[0]

            result[
                label
            ] = (
                float(x),
                float(y),
            )

    return result


# ============================================================
# Models
# ============================================================

def create_detector_model():
    return (
        ssdlite320_mobilenet_v3_large(
            weights=None,
            weights_backbone=None,
            num_classes=DETECTOR_NUM_CLASSES,
        )
    )


def load_detector(
    model_path,
):
    model = create_detector_model()

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
    )

    if (
        isinstance(
            checkpoint,
            dict,
        )
        and "model_state_dict"
        in checkpoint
    ):
        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif (
        isinstance(
            checkpoint,
            dict,
        )
        and "state_dict"
        in checkpoint
    ):
        state_dict = checkpoint[
            "state_dict"
        ]

    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model.to(
        DEVICE
    )

    model.eval()

    return model


def load_keypoint_model(
    model_path,
):
    checkpoint = torch.load(
        model_path,
        map_location="cpu",
    )

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise RuntimeError(
            "Unexpected keypoint checkpoint "
            "format"
        )

    checkpoint_names = checkpoint.get(
        "keypoint_names",
        KEYPOINT_NAMES,
    )

    input_size = checkpoint.get(
        "input_size",
        KP_INPUT_SIZE,
    )

    heatmap_size = checkpoint.get(
        "heatmap_size",
        KP_HEATMAP_SIZE,
    )

    if list(
        checkpoint_names
    ) != list(
        KEYPOINT_NAMES
    ):
        raise RuntimeError(
            "Keypoint name/order mismatch:\n"
            f"  checkpoint="
            f"{checkpoint_names}\n"
            f"  current   ="
            f"{KEYPOINT_NAMES}"
        )

    if int(
        input_size
    ) != int(
        KP_INPUT_SIZE
    ):
        raise RuntimeError(
            "INPUT_SIZE mismatch: "
            f"checkpoint={input_size}, "
            f"current={KP_INPUT_SIZE}"
        )

    if int(
        heatmap_size
    ) != int(
        KP_HEATMAP_SIZE
    ):
        raise RuntimeError(
            "HEATMAP_SIZE mismatch: "
            f"checkpoint={heatmap_size}, "
            f"current={KP_HEATMAP_SIZE}"
        )

    model = MugKeypointHeatmapNet(
        num_keypoints=len(
            KEYPOINT_NAMES
        ),
        pretrained=False,
    )

    if "model_state_dict" not in checkpoint:
        raise RuntimeError(
            "Keypoint checkpoint does not "
            "contain model_state_dict"
        )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.to(
        DEVICE
    )

    model.eval()

    return (
        model,
        checkpoint,
    )


# ============================================================
# Detection
# ============================================================

@torch.inference_mode()
def detect_mug(
    detector,
    image_bgr,
    score_threshold,
):
    rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    tensor = (
        torch.from_numpy(
            rgb
        )
        .permute(
            2,
            0,
            1,
        )
        .float()
        / 255.0
    )

    output = detector(
        [
            tensor.to(
                DEVICE
            )
        ]
    )[0]

    boxes = (
        output[
            "boxes"
        ]
        .detach()
        .cpu()
        .numpy()
    )

    labels = (
        output[
            "labels"
        ]
        .detach()
        .cpu()
        .numpy()
    )

    scores = (
        output[
            "scores"
        ]
        .detach()
        .cpu()
        .numpy()
    )

    best_box = None
    best_score = -1.0

    for (
        box,
        label,
        score,
    ) in zip(
        boxes,
        labels,
        scores,
    ):
        if int(
            label
        ) != MUG_CLASS_ID:
            continue

        if float(
            score
        ) < score_threshold:
            continue

        if float(
            score
        ) > best_score:
            best_box = np.asarray(
                box,
                dtype=np.float64,
            )

            best_score = float(
                score
            )

    return (
        best_box,
        best_score,
    )


# ============================================================
# Canonical fixed 420x420 crop
# ============================================================

def make_keypoint_crop_box(
    bbox_xyxy,
    image_width,
    image_height,
):
    """
    Fixed canonical keypoint crop.

    Rule:
      - center on detector bbox
      - fixed 420x420 in ORIGINAL-image pixels
      - no bbox-dependent scaling
      - if crop would cross an image boundary,
        shift the whole 420x420 window inward
      - do not shrink crop
      - do not pad crop

    This must match real fine-tuning data generation.
    """

    if (
        image_width
        < CANONICAL_CROP_SIZE
        or image_height
        < CANONICAL_CROP_SIZE
    ):
        raise RuntimeError(
            "Original image is smaller than "
            "canonical crop:\n"
            f"  image: "
            f"{image_width}x"
            f"{image_height}\n"
            f"  crop : "
            f"{CANONICAL_CROP_SIZE}x"
            f"{CANONICAL_CROP_SIZE}"
        )

    xmin, ymin, xmax, ymax = [
        float(v)
        for v in bbox_xyxy
    ]

    cx = (
        xmin + xmax
    ) / 2.0

    cy = (
        ymin + ymax
    ) / 2.0

    half = (
        CANONICAL_CROP_SIZE
        / 2.0
    )

    left = int(
        round(
            cx - half
        )
    )

    top = int(
        round(
            cy - half
        )
    )

    max_left = (
        image_width
        - CANONICAL_CROP_SIZE
    )

    max_top = (
        image_height
        - CANONICAL_CROP_SIZE
    )

    left = int(
        np.clip(
            left,
            0,
            max_left,
        )
    )

    top = int(
        np.clip(
            top,
            0,
            max_top,
        )
    )

    right = (
        left
        + CANONICAL_CROP_SIZE
    )

    bottom = (
        top
        + CANONICAL_CROP_SIZE
    )

    return (
        left,
        top,
        right,
        bottom,
    )


def extract_keypoint_crop(
    image_bgr,
    crop_box,
):
    left, top, right, bottom = (
        crop_box
    )

    crop = image_bgr[
        top:bottom,
        left:right,
    ].copy()

    crop_h, crop_w = (
        crop.shape[
            :2
        ]
    )

    if (
        crop_w
        != CANONICAL_CROP_SIZE
        or crop_h
        != CANONICAL_CROP_SIZE
    ):
        raise RuntimeError(
            "Canonical crop extraction "
            "failed:\n"
            f"  crop_box: "
            f"{crop_box}\n"
            f"  result  : "
            f"{crop_w}x"
            f"{crop_h}"
        )

    return crop


# ============================================================
# Keypoint inference
# ============================================================

def heatmap_argmax(
    heatmaps,
):
    b, k, h, w = (
        heatmaps.shape
    )

    flat = heatmaps.reshape(
        b,
        k,
        -1,
    )

    peaks, indices = torch.max(
        flat,
        dim=2,
    )

    y = torch.div(
        indices,
        w,
        rounding_mode="floor",
    )

    x = (
        indices
        % w
    )

    points = torch.stack(
        [
            x.float(),
            y.float(),
        ],
        dim=2,
    )

    return (
        points,
        peaks,
    )


@torch.inference_mode()
def infer_keypoints(
    model,
    crop_bgr,
):
    crop_h, crop_w = (
        crop_bgr.shape[
            :2
        ]
    )

    if (
        crop_w
        != CANONICAL_CROP_SIZE
        or crop_h
        != CANONICAL_CROP_SIZE
    ):
        raise RuntimeError(
            "Unexpected keypoint crop size:\n"
            f"  got   : "
            f"{crop_w}x"
            f"{crop_h}\n"
            f"  expect: "
            f"{CANONICAL_CROP_SIZE}x"
            f"{CANONICAL_CROP_SIZE}"
        )

    # Normally 420 -> 420.
    # Keep this explicit so inference follows dataset.py
    # if INPUT_SIZE is intentionally changed later.
    if (
        crop_w != KP_INPUT_SIZE
        or crop_h != KP_INPUT_SIZE
    ):
        input_bgr = cv2.resize(
            crop_bgr,
            (
                KP_INPUT_SIZE,
                KP_INPUT_SIZE,
            ),
            interpolation=cv2.INTER_LINEAR,
        )
    else:
        input_bgr = crop_bgr

    rgb = cv2.cvtColor(
        input_bgr,
        cv2.COLOR_BGR2RGB,
    )

    tensor = (
        torch.from_numpy(
            rgb
        )
        .permute(
            2,
            0,
            1,
        )
        .float()
        / 255.0
    )

    tensor = (
        tensor
        .unsqueeze(0)
        .to(
            DEVICE
        )
    )

    heatmaps = model(
        tensor
    )

    points_hm_batch, peaks_batch = (
        heatmap_argmax(
            heatmaps
        )
    )

    points_hm = (
        points_hm_batch[
            0
        ]
        .detach()
        .cpu()
        .numpy()
    )

    peaks = (
        peaks_batch[
            0
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float64
        )
    )

    scale = (
        KP_INPUT_SIZE
        / KP_HEATMAP_SIZE
    )

    points_input = (
        points_hm
        * scale
    )

    return (
        input_bgr,
        points_input.astype(
            np.float64
        ),
        peaks,
    )


def input_to_original(
    points_input,
    crop_box,
):
    left, top, right, bottom = (
        crop_box
    )

    crop_w = float(
        right - left
    )

    crop_h = float(
        bottom - top
    )

    points_original = (
        points_input
        .astype(
            np.float64
        )
        .copy()
    )

    points_original[
        :,
        0
    ] = (
        left
        + points_input[
            :,
            0
        ]
        / KP_INPUT_SIZE
        * crop_w
    )

    points_original[
        :,
        1
    ] = (
        top
        + points_input[
            :,
            1
        ]
        / KP_INPUT_SIZE
        * crop_h
    )

    return points_original


# ============================================================
# PnP
# ============================================================

def reprojection_errors(
    object_points,
    image_points,
    rvec,
    tvec,
):
    projected, _ = (
        cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            CAMERA_MATRIX,
            DIST_COEFFS,
        )
    )

    projected = projected.reshape(
        -1,
        2,
    )

    return np.linalg.norm(
        projected
        - image_points,
        axis=1,
    )


def rmse(
    values,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if len(
        values
    ) == 0:
        return float(
            "inf"
        )

    return float(
        np.sqrt(
            np.mean(
                values ** 2
            )
        )
    )


def solve_pose(
    image_points,
):
    object_points = (
        OBJECT_POINTS
    )

    cv2.setRNGSeed(
        0
    )

    (
        success,
        rvec,
        tvec,
        inliers,
    ) = cv2.solvePnPRansac(
        object_points,
        image_points,
        CAMERA_MATRIX,
        DIST_COEFFS,
        flags=cv2.SOLVEPNP_EPNP,
        reprojectionError=RANSAC_REPROJ_ERROR,
        iterationsCount=RANSAC_ITERATIONS,
        confidence=0.99,
    )

    if (
        not success
        or rvec is None
        or tvec is None
    ):
        return {
            "status":
                "RANSAC_FAIL",

            "accepted":
                False,

            "rvec":
                None,

            "tvec":
                None,

            "initial_inliers":
                [],

            "final_inliers":
                [],

            "errors":
                None,

            "used_rmse":
                None,

            "all_rmse":
                None,

            "top_count":
                "",

            "bottom_count":
                "",

            "reason":
                "solvePnPRansac failed",
        }

    if inliers is None:
        current_indices = np.arange(
            len(
                object_points
            ),
            dtype=np.int32,
        )
    else:
        current_indices = (
            inliers
            .reshape(
                -1
            )
            .astype(
                np.int32
            )
        )

    initial_indices = (
        current_indices.copy()
    )

    for _ in range(
        MAX_REFINE_ROUNDS
    ):
        if len(
            current_indices
        ) < 4:
            break

        (
            ok,
            rvec,
            tvec,
        ) = cv2.solvePnP(
            object_points[
                current_indices
            ],
            image_points[
                current_indices
            ],
            CAMERA_MATRIX,
            DIST_COEFFS,
            rvec,
            tvec,
            True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not ok:
            break

        errors = reprojection_errors(
            object_points,
            image_points,
            rvec,
            tvec,
        )

        new_indices = np.where(
            errors
            <= FINAL_INLIER_ERROR
        )[0].astype(
            np.int32
        )

        if np.array_equal(
            new_indices,
            current_indices,
        ):
            current_indices = (
                new_indices
            )
            break

        current_indices = (
            new_indices
        )

    if len(
        current_indices
    ) >= 4:
        (
            ok,
            rvec,
            tvec,
        ) = cv2.solvePnP(
            object_points[
                current_indices
            ],
            image_points[
                current_indices
            ],
            CAMERA_MATRIX,
            DIST_COEFFS,
            rvec,
            tvec,
            True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not ok:
            return {
                "status":
                    "REFINE_FAIL",

                "accepted":
                    False,

                "rvec":
                    rvec,

                "tvec":
                    tvec,

                "initial_inliers":
                    initial_indices.tolist(),

                "final_inliers":
                    current_indices.tolist(),

                "errors":
                    None,

                "used_rmse":
                    None,

                "all_rmse":
                    None,

                "top_count":
                    "",

                "bottom_count":
                    "",

                "reason":
                    "final solvePnP refine failed",
            }

    errors = reprojection_errors(
        object_points,
        image_points,
        rvec,
        tvec,
    )

    final_indices = np.where(
        errors
        <= FINAL_INLIER_ERROR
    )[0].astype(
        np.int32
    )

    used_rmse = rmse(
        errors[
            final_indices
        ]
    )

    all_rmse = rmse(
        errors
    )

    top_count = sum(
        int(i)
        in TOP_INDICES
        for i in final_indices
    )

    bottom_count = sum(
        int(i)
        in BOTTOM_INDICES
        for i in final_indices
    )

    reasons = []

    if float(
        tvec[
            2,
            0
        ]
    ) <= 0.0:
        reasons.append(
            "z<=0"
        )

    if len(
        final_indices
    ) < MIN_INLIERS:
        reasons.append(
            f"inliers<"
            f"{MIN_INLIERS}"
        )

    if (
        top_count
        < MIN_TOP_INLIERS
    ):
        reasons.append(
            f"top<"
            f"{MIN_TOP_INLIERS}"
        )

    if (
        bottom_count
        < MIN_BOTTOM_INLIERS
    ):
        reasons.append(
            f"bottom<"
            f"{MIN_BOTTOM_INLIERS}"
        )

    if (
        used_rmse
        > MAX_USED_RMSE
    ):
        reasons.append(
            f"used_rmse>"
            f"{MAX_USED_RMSE:.1f}"
        )

    if (
        all_rmse
        > MAX_ALL_RMSE
    ):
        reasons.append(
            f"all_rmse>"
            f"{MAX_ALL_RMSE:.1f}"
        )

    accepted = (
        len(
            reasons
        ) == 0
    )

    return {
        "status":
            "OK"
            if accepted
            else "NG",

        "accepted":
            accepted,

        "rvec":
            rvec,

        "tvec":
            tvec,

        "initial_inliers":
            initial_indices.tolist(),

        "final_inliers":
            final_indices.tolist(),

        "errors":
            errors,

        "used_rmse":
            used_rmse,

        "all_rmse":
            all_rmse,

        "top_count":
            top_count,

        "bottom_count":
            bottom_count,

        "reason":
            ""
            if accepted
            else ",".join(
                reasons
            ),
    }


# ============================================================
# Drawing
# ============================================================

def draw_detection(
    image_bgr,
    bbox,
    crop_box,
    score,
):
    result = (
        image_bgr.copy()
    )

    x1, y1, x2, y2 = [
        int(
            round(
                float(v)
            )
        )
        for v in bbox
    ]

    cv2.rectangle(
        result,
        (
            x1,
            y1,
        ),
        (
            x2,
            y2,
        ),
        (
            255,
            255,
            0,
        ),
        2,
    )

    (
        left,
        top,
        right,
        bottom,
    ) = crop_box

    cv2.rectangle(
        result,
        (
            left,
            top,
        ),
        (
            right,
            bottom,
        ),
        (
            180,
            180,
            180,
        ),
        2,
    )

    cv2.putText(
        result,
        f"mug={score:.3f}",
        (
            max(
                0,
                x1,
            ),
            max(
                20,
                y1 - 8,
            ),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (
            255,
            255,
            0,
        ),
        2,
        cv2.LINE_AA,
    )

    return result


def draw_keypoints(
    image_bgr,
    points,
    conf=None,
    gt_points=None,
    offset=(
        0,
        0,
    ),
):
    result = (
        image_bgr.copy()
    )

    ox, oy = offset

    for (
        i,
        name,
    ) in enumerate(
        KEYPOINT_NAMES
    ):
        x = int(
            round(
                float(
                    points[
                        i,
                        0
                    ]
                    - ox
                )
            )
        )

        y = int(
            round(
                float(
                    points[
                        i,
                        1
                    ]
                    - oy
                )
            )
        )

        cv2.drawMarker(
            result,
            (
                x,
                y,
            ),
            (
                255,
                0,
                255,
            ),
            markerType=cv2.MARKER_CROSS,
            markerSize=13,
            thickness=2,
            line_type=cv2.LINE_AA,
        )

        label = name

        if conf is not None:
            label += (
                f" "
                f"{conf[i]:.3f}"
            )

        cv2.putText(
            result,
            label,
            (
                x + 5,
                y - 5,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (
                255,
                0,
                255,
            ),
            1,
            cv2.LINE_AA,
        )

    if gt_points:
        for (
            name,
            (
                gx,
                gy,
            ),
        ) in gt_points.items():

            x = int(
                round(
                    gx - ox
                )
            )

            y = int(
                round(
                    gy - oy
                )
            )

            cv2.circle(
                result,
                (
                    x,
                    y,
                ),
                6,
                (
                    0,
                    255,
                    0,
                ),
                2,
                cv2.LINE_AA,
            )

    return result


def draw_pnp(
    image_bgr,
    image_points,
    pose,
):
    result = (
        image_bgr.copy()
    )

    final_inliers = set(
        pose.get(
            "final_inliers",
            [],
        )
    )

    initial_inliers = set(
        pose.get(
            "initial_inliers",
            [],
        )
    )

    for (
        i,
        name,
    ) in enumerate(
        KEYPOINT_NAMES
    ):
        p = (
            int(
                round(
                    image_points[
                        i,
                        0
                    ]
                )
            ),
            int(
                round(
                    image_points[
                        i,
                        1
                    ]
                )
            ),
        )

        if i in final_inliers:
            color = (
                255,
                0,
                255,
            )

        elif i in initial_inliers:
            color = (
                0,
                255,
                255,
            )

        else:
            color = (
                0,
                128,
                255,
            )

        cv2.circle(
            result,
            p,
            5,
            color,
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            result,
            name,
            (
                p[0] + 5,
                p[1] - 5,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    status_text = pose[
        "status"
    ]

    if (
        pose.get(
            "used_rmse"
        )
        is not None
    ):
        status_text += (
            f" inliers="
            f"{len(pose['final_inliers'])}"
            f" rmse="
            f"{pose['used_rmse']:.2f}"
        )

    if pose.get(
        "reason"
    ):
        status_text += (
            f" "
            f"{pose['reason']}"
        )

    cv2.putText(
        result,
        status_text,
        (
            10,
            28,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (
            255,
            255,
            255,
        ),
        2,
        cv2.LINE_AA,
    )

    rvec = pose.get(
        "rvec"
    )

    tvec = pose.get(
        "tvec"
    )

    if (
        rvec is not None
        and tvec is not None
    ):
        axis_length = 0.05

        axis_object = np.asarray(
            [
                [
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    axis_length,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    axis_length,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    axis_length,
                ],
            ],
            dtype=np.float64,
        )

        projected, _ = (
            cv2.projectPoints(
                axis_object,
                rvec,
                tvec,
                CAMERA_MATRIX,
                DIST_COEFFS,
            )
        )

        pts = projected.reshape(
            -1,
            2,
        )

        if np.isfinite(
            pts
        ).all():
            pts = np.round(
                pts
            ).astype(
                np.int32
            )

            origin = tuple(
                pts[
                    0
                ]
            )

            cv2.line(
                result,
                origin,
                tuple(
                    pts[
                        1
                    ]
                ),
                (
                    0,
                    0,
                    255,
                ),
                3,
                cv2.LINE_AA,
            )

            cv2.line(
                result,
                origin,
                tuple(
                    pts[
                        2
                    ]
                ),
                (
                    0,
                    255,
                    0,
                ),
                3,
                cv2.LINE_AA,
            )

            cv2.line(
                result,
                origin,
                tuple(
                    pts[
                        3
                    ]
                ),
                (
                    255,
                    0,
                    0,
                ),
                3,
                cv2.LINE_AA,
            )

    return result


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--frames-root",
        type=Path,
        default=DEFAULT_FRAMES_ROOT,
    )

    parser.add_argument(
        "--detector-model",
        type=Path,
        default=DEFAULT_DETECTOR_PATH,
    )

    parser.add_argument(
        "--keypoint-model",
        type=Path,
        default=DEFAULT_KEYPOINT_MODEL_PATH,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--labelme-root",
        type=Path,
        default=DEFAULT_LABELME_ROOT,
    )

    parser.add_argument(
        "--det-score",
        type=float,
        default=DETECTION_SCORE_THRESHOLD,
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    print(
        f"Device         : "
        f"{DEVICE}"
    )

    print(
        f"Frames root    : "
        f"{args.frames_root}"
    )

    print(
        f"Detector       : "
        f"{args.detector_model}"
    )

    print(
        f"Keypoint model : "
        f"{args.keypoint_model}"
    )

    print(
        f"Canonical crop : "
        f"{CANONICAL_CROP_SIZE}x"
        f"{CANONICAL_CROP_SIZE}"
    )

    print(
        f"KP input       : "
        f"{KP_INPUT_SIZE}x"
        f"{KP_INPUT_SIZE}"
    )

    print(
        f"KP heatmap     : "
        f"{KP_HEATMAP_SIZE}x"
        f"{KP_HEATMAP_SIZE}"
    )

    detector = load_detector(
        args.detector_model
    )

    (
        keypoint_model,
        kp_checkpoint,
    ) = load_keypoint_model(
        args.keypoint_model
    )

    print(
        f"KP checkpoint  : "
        f"epoch="
        f"{kp_checkpoint.get('epoch', -1)}"
    )

    labelme_index = (
        build_labelme_index(
            args.labelme_root
        )
    )

    image_paths = (
        collect_images(
            args.frames_root
        )
    )

    if args.max_images is not None:
        image_paths = image_paths[
            :args.max_images
        ]

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        args.output_dir
        / "pipeline_log.csv"
    )

    csv_header = [
        "image",
        "mug_score",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "crop_left",
        "crop_top",
        "crop_right",
        "crop_bottom",
        "pnp_status",
        "pnp_reason",
        "pnp_final_inliers",
        "pnp_top_inliers",
        "pnp_bottom_inliers",
        "pnp_used_rmse",
        "pnp_all_rmse",
        "detector_ms",
        "crop_ms",
        "keypoint_ms",
        "pnp_ms",
        "total_ms",
        "fps",
    ]

    for name in KEYPOINT_NAMES:
        csv_header += [
            f"{name}_x",
            f"{name}_y",
            f"{name}_conf",
        ]

    processed = 0
    no_detection = 0

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(
            f
        )

        writer.writerow(
            csv_header
        )

        for image_path in image_paths:

            image = cv2.imread(
                str(
                    image_path
                ),
                cv2.IMREAD_COLOR,
            )

            if image is None:
                print(
                    f"[WARN] unreadable: "
                    f"{image_path}"
                )
                continue

            image_h, image_w = (
                image.shape[
                    :2
                ]
            )

            # Timing excludes image loading, visualization, PNG saving,
            # and CSV writing. CUDA is synchronized around GPU stages so
            # perf_counter measures completed work rather than queued work.
            synchronize_device()
            t_total_start = time.perf_counter()

            synchronize_device()
            t0 = time.perf_counter()
            bbox, score = detect_mug(
                detector,
                image,
                args.det_score,
            )
            synchronize_device()
            detector_ms = (
                time.perf_counter() - t0
            ) * 1000.0

            if bbox is None:
                no_detection += 1
                continue

            t0 = time.perf_counter()
            crop_box = (
                make_keypoint_crop_box(
                    bbox,
                    image_w,
                    image_h,
                )
            )

            crop = extract_keypoint_crop(
                image,
                crop_box,
            )
            crop_ms = (
                time.perf_counter() - t0
            ) * 1000.0

            synchronize_device()
            t0 = time.perf_counter()
            (
                crop_input,
                points_input,
                kp_conf,
            ) = infer_keypoints(
                keypoint_model,
                crop,
            )
            synchronize_device()
            keypoint_ms = (
                time.perf_counter() - t0
            ) * 1000.0

            points_original = (
                input_to_original(
                    points_input,
                    crop_box,
                )
            )

            # PnP receives ONLY original-camera coordinates.
            t0 = time.perf_counter()
            pose = solve_pose(
                points_original
            )
            pnp_ms = (
                time.perf_counter() - t0
            ) * 1000.0

            synchronize_device()
            total_ms = (
                time.perf_counter() - t_total_start
            ) * 1000.0

            fps = (
                1000.0 / total_ms
                if total_ms > 0.0
                else 0.0
            )

            relative = (
                image_path.relative_to(
                    args.frames_root
                )
            )

            group_dir = (
                args.output_dir
                / relative.parent
            )

            group_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            stem = (
                image_path.stem
            )

            json_path = (
                labelme_index.get(
                    image_path.name
                )
            )

            gt_points = (
                read_labelme_keypoints(
                    json_path
                )
                if json_path is not None
                else {}
            )

            detection_vis = (
                draw_detection(
                    image,
                    bbox,
                    crop_box,
                    score,
                )
            )

            cv2.imwrite(
                str(
                    group_dir
                    / f"{stem}_detection.png"
                ),
                detection_vis,
            )

            cv2.imwrite(
                str(
                    group_dir
                    / f"{stem}_crop.png"
                ),
                crop,
            )

            crop_kp_vis = draw_keypoints(
                crop_input,
                points_input,
                conf=kp_conf,
            )

            cv2.imwrite(
                str(
                    group_dir
                    / (
                        f"{stem}_"
                        "keypoints_crop.png"
                    )
                ),
                crop_kp_vis,
            )

            original_kp_vis = (
                draw_keypoints(
                    image,
                    points_original,
                    conf=kp_conf,
                    gt_points=gt_points,
                )
            )

            cv2.imwrite(
                str(
                    group_dir
                    / (
                        f"{stem}_"
                        "keypoints_original.png"
                    )
                ),
                original_kp_vis,
            )

            pnp_vis = draw_pnp(
                image,
                points_original,
                pose,
            )

            cv2.imwrite(
                str(
                    group_dir
                    / f"{stem}_pnp.png"
                ),
                pnp_vis,
            )

            row = [
                str(
                    relative
                ),

                f"{score:.8f}",

                *[
                    f"{float(v):.6f}"
                    for v in bbox
                ],

                *[
                    int(v)
                    for v in crop_box
                ],

                pose[
                    "status"
                ],

                pose.get(
                    "reason",
                    "",
                ),

                len(
                    pose.get(
                        "final_inliers",
                        [],
                    )
                ),

                pose.get(
                    "top_count",
                    "",
                ),

                pose.get(
                    "bottom_count",
                    "",
                ),

                (
                    ""
                    if pose.get(
                        "used_rmse"
                    ) is None
                    else
                    f"{pose['used_rmse']:.6f}"
                ),

                (
                    ""
                    if pose.get(
                        "all_rmse"
                    ) is None
                    else
                    f"{pose['all_rmse']:.6f}"
                ),

                f"{detector_ms:.6f}",
                f"{crop_ms:.6f}",
                f"{keypoint_ms:.6f}",
                f"{pnp_ms:.6f}",
                f"{total_ms:.6f}",
                f"{fps:.6f}",
            ]

            for i in range(
                len(
                    KEYPOINT_NAMES
                )
            ):
                row += [
                    f"{points_original[i, 0]:.6f}",
                    f"{points_original[i, 1]:.6f}",
                    f"{kp_conf[i]:.8f}",
                ]

            writer.writerow(
                row
            )

            processed += 1

            if (
                processed == 1
                or processed % 50 == 0
            ):
                print(
                    f"Processed: "
                    f"{processed}/"
                    f"{len(image_paths)}"
                )

    print()
    print(
        "=" * 60
    )
    print(
        "UNIFIED REAL PIPELINE COMPLETE"
    )
    print(
        "=" * 60
    )
    print(
        f"Images       : "
        f"{len(image_paths)}"
    )
    print(
        f"Processed    : "
        f"{processed}"
    )
    print(
        f"No detection : "
        f"{no_detection}"
    )
    print(
        f"Output       : "
        f"{args.output_dir}"
    )
    print(
        f"CSV log      : "
        f"{csv_path}"
    )


if __name__ == "__main__":
    main()

