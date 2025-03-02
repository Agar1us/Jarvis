import torch
from omniparser.utils import get_caption_model_processor, check_ocr_box, get_som_labeled_img
from PIL import Image
import io
import base64
from typing import Dict
from ultralytics import YOLO
import pyautogui

CONFIG = {'BOX_THRESHOLD': 0.05, 'yolo_model_path': 'model.pt', 'caption_model_path': 'icon_caption'} 

class Omniparser(object):
    def __init__(self, config: Dict):
        self.config = config
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.som_model = YOLO(config['yolo_model_path'])
        self.caption_model_processor = get_caption_model_processor(model_name_or_path=config['caption_model_path'], device=device)

    def parse(self, image_base64: str):
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        
        box_overlay_ratio = max(image.size) / 3200
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }

        (text, ocr_bbox), _ = check_ocr_box(image, display_img=False, output_bb_format='xyxy', easyocr_args={'text_threshold': 0.8})
        dino_labled_img, parsed_content_list = get_som_labeled_img(
            image, 
            self.som_model, 
            BOX_TRESHOLD = self.config['BOX_THRESHOLD'], 
            output_coord_in_ratio=False, 
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config, 
            caption_model_processor=self.caption_model_processor, 
            ocr_text=text,
            use_local_semantics=True, 
            iou_threshold=0.7, 
            scale_img=False, 
            batch_size=128)

        return dino_labled_img, parsed_content_list
    
model = Omniparser(CONFIG)

def make_screenshot():
    screenshot = pyautogui.screenshot()
    buffered = io.BytesIO()
    screenshot.save(buffered, format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
    labeled_img, description = model.parse(encoded_image)
    return labeled_img, str(description)
