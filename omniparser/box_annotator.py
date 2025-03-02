from typing import List, Optional, Union, Tuple, Any
from enum import Enum

import cv2
import numpy as np

from supervision.detection.core import Detections
from supervision.draw.color import Color, ColorPalette

class BoxAnnotator:
    """
    A class for drawing bounding boxes on an image using detections provided.

    Attributes:
        color (Union[Color, ColorPalette]): The color to draw the bounding box,
            can be a single color or a color palette
        thickness (int): The thickness of the bounding box lines, default is 2
        text_color (Color): The color of the text on the bounding box, default is white
        text_scale (float): The scale of the text on the bounding box, default is 0.5
        text_thickness (int): The thickness of the text on the bounding box,
            default is 1
        text_padding (int): The padding around the text on the bounding box,
            default is 5

    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.DEFAULT,
        thickness: int = 2, # 1 for seeclick 2 for mind2web and 3 for demo
        text_color: Color = Color.BLACK,
        text_scale: float = 0.3, # 0.8 for mobile/web, 0.3 for desktop # 0.4 for mind2web
        text_thickness: int = 1, #1, # 2 for demo
        text_padding: int = 10,
        avoid_overlap: bool = True,
    ):
        self.color: Union[Color, ColorPalette] = color
        self.thickness: int = thickness
        self.text_color: Color = text_color
        self.text_scale: float = text_scale
        self.text_thickness: int = text_thickness
        self.text_padding: int = text_padding
        self.avoid_overlap: bool = avoid_overlap

    def annotate(
        self,
        scene: np.ndarray,
        detections: Detections,
        labels: Optional[List[str]] = None,
        skip_label: bool = False,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Draws bounding boxes on the frame using the detections provided.

        Args:
            scene (np.ndarray): The image on which the bounding boxes will be drawn
            detections (Detections): The detections for which the
                bounding boxes will be drawn
            labels (Optional[List[str]]): An optional list of labels
                corresponding to each detection. If `labels` are not provided,
                corresponding `class_id` will be used as label.
            skip_label (bool): Is set to `True`, skips bounding box label annotation.
        Returns:
            np.ndarray: The image with the bounding boxes drawn on it
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i].astype(int)
            class_id = (
                detections.class_id[i] if detections.class_id is not None else None
            )
            idx = class_id if class_id is not None else i
            color = (
                self.color.by_idx(idx)
                if isinstance(self.color, ColorPalette)
                else self.color
            )
            cv2.rectangle(
                img=scene,
                pt1=(x1, y1),
                pt2=(x2, y2),
                color=color.as_bgr(),
                thickness=self.thickness,
            )
            if skip_label:
                continue

            text = (
                f"{class_id}"
                if (labels is None or len(detections) != len(labels))
                else labels[i]
            )

            text_width, text_height = cv2.getTextSize(
                text=text,
                fontFace=font,
                fontScale=self.text_scale,
                thickness=self.text_thickness,
            )[0]

            if not self.avoid_overlap:
                text_x = x1 + self.text_padding
                text_y = y1 - self.text_padding

                text_background_x1 = x1
                text_background_y1 = y1 - 2 * self.text_padding - text_height

                text_background_x2 = x1 + 2 * self.text_padding + text_width
                text_background_y2 = y1
            else:
                text_x, text_y, text_background_x1, text_background_y1, text_background_x2, text_background_y2 = get_optimal_label_pos(self.text_padding, text_width, text_height, x1, y1, x2, y2, detections, image_size)

            cv2.rectangle(
                img=scene,
                pt1=(text_background_x1, text_background_y1),
                pt2=(text_background_x2, text_background_y2),
                color=color.as_bgr(),
                thickness=cv2.FILLED,
            )
            box_color = color.as_rgb()
            luminance = 0.299 * box_color[0] + 0.587 * box_color[1] + 0.114 * box_color[2]
            text_color = (0,0,0) if luminance > 160 else (255,255,255)
            cv2.putText(
                img=scene,
                text=text,
                org=(text_x, text_y),
                fontFace=font,
                fontScale=self.text_scale,
                color=text_color,
                thickness=self.text_thickness,
                lineType=cv2.LINE_AA,
            )
        return scene
    
def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

def intersection_area(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    return max(0, x2 - x1) * max(0, y2 - y1)

def IoU(box1, box2, return_max=True):
    intersection = intersection_area(box1, box2)
    union = box_area(box1) + box_area(box2) - intersection + 1e-6
    if box_area(box1) > 0 and box_area(box2) > 0:
        ratio1 = intersection / box_area(box1)
        ratio2 = intersection / box_area(box2)
    else:
        ratio1, ratio2 = 0, 0
    if return_max:
        return max(intersection / union, ratio1, ratio2)
    else:
        return intersection / union

class LabelPosition(Enum):
    """Enum for possible label positions relative to the bounding box."""
    TOP_LEFT = "top left"
    TOP_RIGHT = "top right"
    BOTTOM_LEFT = "bottom left"
    BOTTOM_RIGHT = "bottom right"
    OUTER_LEFT = "outer left"
    OUTER_RIGHT = "outer right"

def is_box_out_of_image(
    box: List[float],
    image_size: Tuple[int, int]
) -> bool:
    """
    Check if the box is outside the image boundaries.

    Args:
        box: Bounding box coordinates [x1, y1, x2, y2].
        image_size: Tuple of (width, height) of the image.

    Returns:
        True if the box is out of bounds, False otherwise.
    """
    x1, y1, x2, y2 = box
    img_width, img_height = image_size
    return x1 < 0 or y1 < 0 or x2 > img_width or y2 > img_height

def calculate_label_coordinates(
    position: LabelPosition,
    box: List[float],
    text_width: int,
    text_height: int,
    text_padding: int
) -> Tuple[float, float, List[float]]:
    """
    Calculate text and background box coordinates for a given label position.
    """
    x1, y1, x2, y2 = box
    
    # Увеличим отступы для лучшей читаемости
    horizontal_padding = text_padding
    vertical_padding = text_padding

    match position:
        case LabelPosition.TOP_LEFT:
            text_x = x1 + horizontal_padding
            text_y = y1 - vertical_padding
            background_box = [
                x1,
                y1 - 2 * vertical_padding - text_height,
                x1 + 2 * horizontal_padding + text_width,
                y1
            ]
        case LabelPosition.TOP_RIGHT:
            text_x = x2 - horizontal_padding - text_width
            text_y = y1 - vertical_padding
            background_box = [
                x2 - 2 * horizontal_padding - text_width,
                y1 - 2 * vertical_padding - text_height,
                x2,
                y1
            ]
        case LabelPosition.BOTTOM_LEFT:
            text_x = x1 + horizontal_padding
            text_y = y2 + text_height + vertical_padding
            background_box = [
                x1,
                y2,
                x1 + 2 * horizontal_padding + text_width,
                y2 + 2 * vertical_padding + text_height
            ]
        case LabelPosition.BOTTOM_RIGHT:
            text_x = x2 - horizontal_padding - text_width
            text_y = y2 + text_height + vertical_padding
            background_box = [
                x2 - 2 * horizontal_padding - text_width,
                y2,
                x2,
                y2 + 2 * vertical_padding + text_height
            ]
        case LabelPosition.OUTER_LEFT:
            text_x = x1 - horizontal_padding - text_width
            text_y = y1 + vertical_padding + text_height
            background_box = [
                x1 - 2 * horizontal_padding - text_width,
                y1,
                x1,
                y1 + 2 * vertical_padding + text_height
            ]
        case LabelPosition.OUTER_RIGHT:
            text_x = x2 + horizontal_padding
            text_y = y1 + vertical_padding + text_height
            background_box = [
                x2,
                y1,
                x2 + 2 * horizontal_padding + text_width,
                y1 + 2 * vertical_padding + text_height
            ]
        case _:
            raise ValueError(f"Unsupported label position: {position}")

    return text_x, text_y, background_box

def get_optimal_label_pos(
    text_padding: int,
    text_width: int,
    text_height: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    detections: Any,
    image_size: Tuple[int, int],
) -> Tuple[float, float, float, float, float, float]:
    """
    Find the optimal position for a text label, avoiding overlap with detections and image boundaries.
    """
    box = [x1, y1, x2, y2]
    default_position = LabelPosition.OUTER_RIGHT
    last_valid_coords = None
    min_overlap = float('inf')
    best_coords = None

    # Попробуем все возможные позиции
    for position in LabelPosition:
        text_x, text_y, background_box = calculate_label_coordinates(
            position, box, text_width, text_height, text_padding
        )

        # Проверяем перекрытие с другими боксами
        overlap_count = 0
        for i in range(len(detections)):
            detection = detections.xyxy[i].astype(int)
            if IoU(background_box, detection) > 0.1:
                overlap_count += 1

        # Проверяем выход за границы изображения
        if not is_box_out_of_image(background_box, image_size):
            if overlap_count <= min_overlap:
                min_overlap = overlap_count
                best_coords = (text_x, text_y, *background_box)

        # Сохраняем координаты для позиции по умолчанию
        if position == default_position:
            last_valid_coords = (text_x, text_y, *background_box)

    # Возвращаем лучшую найденную позицию или позицию по умолчанию
    return tuple(map(int, best_coords)) if best_coords is not None else tuple(map(int, last_valid_coords))