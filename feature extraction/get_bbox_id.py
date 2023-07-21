import torch
from typing import Tuple, List
from detectron2.structures import Boxes, Instances
from detectron2.layers import (
    ShapeSpec,
    batched_nms,
    cat,
    ciou_loss,
    cross_entropy,
    diou_loss,
    nonzero_tuple,
)

from torch.nn import functional as F

def inference( predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances], box2box_transform, test_topk_per_image = 50):
    """
    Args:
        predictions: return values of :meth:`forward()`.
        proposals (list[Instances]): proposals that match the features that were
            used to compute predictions. The ``proposal_boxes`` field is expected.

    Returns:
        list[Instances]: same as `fast_rcnn_inference`.
        list[Tensor]: same as `fast_rcnn_inference`.
    """
    test_score_thresh = 0.0
    test_nms_thresh = 0.5
    boxes = predict_boxes(predictions, proposals, box2box_transform)
    scores = predict_probs(predictions, proposals)
    image_shapes = [x.image_size for x in proposals]
    return fast_rcnn_inference(
        boxes,
        scores,
        image_shapes,
        test_score_thresh,
        test_nms_thresh,
        test_topk_per_image,
    )

def predict_boxes(predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances], box2box_transform):
    """
    Args:
        predictions: return values of :meth:`forward()`.
        proposals (list[Instances]): proposals that match the features that were
            used to compute predictions. The ``proposal_boxes`` field is expected.

    Returns:
        list[Tensor]:
            A list of Tensors of predicted class-specific or class-agnostic boxes
            for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
            the number of proposals for image i and B is the box dimension (4 or 5)
    """
    if not len(proposals):
        return []
    _, proposal_deltas = predictions
    num_prop_per_image = [len(p) for p in proposals]
    proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
    predict_boxes = box2box_transform.apply_deltas(
        proposal_deltas,
        proposal_boxes,
    )  # Nx(KxB)
    return predict_boxes.split(num_prop_per_image)

def predict_probs(predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]):
    """
    Args:
        predictions: return values of :meth:`forward()`.
        proposals (list[Instances]): proposals that match the features that were
            used to compute predictions.

    Returns:
        list[Tensor]:
            A list of Tensors of predicted class probabilities for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of proposals for image i.
    """
    scores, _ = predictions
    num_inst_per_image = [len(p) for p in proposals]
    probs = F.softmax(scores, dim=-1)
    return probs.split(num_inst_per_image, dim=0)


def fast_rcnn_inference(
    boxes: List[torch.Tensor],
    scores: List[torch.Tensor],
    image_shapes: List[Tuple[int, int]],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
):
    """
    Call `fast_rcnn_inference_single_image` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputLayers.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh, topk_per_image
        )
        for scores_per_image, boxes_per_image, image_shape in zip(scores, boxes, image_shapes)
    ]
    return result_per_image

def fast_rcnn_inference_single_image(
    boxes,
    scores,
    image_shape: Tuple[int, int],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]

    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]

    # 2. Apply NMS for each class independently.
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    return keep