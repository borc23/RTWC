"""Coordinate conversion utilities for YOLO bounding boxes."""


def yolo_to_abs_bbox(yolo_coords, img_width, img_height):
    """Convert YOLO normalized coordinates to absolute pixel coordinates.

    Args:
        yolo_coords: Tuple of (class_id, center_x_norm, center_y_norm, w_norm, h_norm)
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        Tuple of (class_id, xmin, ymin, xmax, ymax) in absolute coordinates
    """
    class_id, center_x_norm, center_y_norm, w_norm, h_norm = yolo_coords
    center_x = center_x_norm * img_width
    center_y = center_y_norm * img_height
    w = w_norm * img_width
    h = h_norm * img_height
    xmin = center_x - w / 2
    ymin = center_y - h / 2
    xmax = center_x + w / 2
    ymax = center_y + h / 2
    return int(class_id), xmin, ymin, xmax, ymax


def abs_bbox_to_yolo(abs_coords, slice_width, slice_height):
    """Convert absolute pixel coordinates to YOLO normalized coordinates.

    Args:
        abs_coords: Tuple of (class_id, xmin, ymin, xmax, ymax) in absolute coordinates
        slice_width: Width of the image/slice in pixels
        slice_height: Height of the image/slice in pixels

    Returns:
        Tuple of (class_id, center_x_norm, center_y_norm, w_norm, h_norm)
    """
    class_id, xmin, ymin, xmax, ymax = abs_coords
    w = xmax - xmin
    h = ymax - ymin
    center_x = xmin + w / 2
    center_y = ymin + h / 2
    center_x_norm = center_x / slice_width
    center_y_norm = center_y / slice_height
    w_norm = w / slice_width
    h_norm = h / slice_height
    return class_id, center_x_norm, center_y_norm, w_norm, h_norm
