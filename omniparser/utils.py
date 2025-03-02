import io
import base64
from PIL import Image
# utility function
import cv2
import numpy as np
# %matplotlib inline
from matplotlib import pyplot as plt
from omniparser.box_annotator import box_area, intersection_area, IoU

from typing import Dict, Optional
import base64

import torch
from typing import Tuple, List, Union
from torchvision.ops import box_convert
from torchvision.transforms import ToPILImage
import supervision as sv
import torchvision.transforms as T
from omniparser.box_annotator import BoxAnnotator 
from transformers import AutoProcessor, AutoModelForCausalLM
import easyocr
reader = easyocr.Reader(['en', 'ru'])

def get_caption_model_processor(model_name_or_path="microsoft/Florence-2-base", device='cpu'):
    processor = AutoProcessor.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float32 if device == 'cpu' else torch.float16, trust_remote_code=True)
    return {'model': model.to(device), 'processor': processor}

@torch.inference_mode()
def get_parsed_content_icon(filtered_boxes, starting_idx, image_source, caption_model_processor, batch_size=128):
    # Number of samples per batch, --> 128 roughly takes 4 GB of GPU memory for florence v2 model
    to_pil = ToPILImage()
    if starting_idx:
        non_ocr_boxes = filtered_boxes[starting_idx:]
    else:
        non_ocr_boxes = filtered_boxes
    croped_pil_image = []
    for i, coord in enumerate(non_ocr_boxes):
        try:
            xmin, xmax = int(coord[0]*image_source.shape[1]), int(coord[2]*image_source.shape[1])
            ymin, ymax = int(coord[1]*image_source.shape[0]), int(coord[3]*image_source.shape[0])
            cropped_image = image_source[ymin:ymax, xmin:xmax, :]
            cropped_image = cv2.resize(cropped_image, (64, 64))
            croped_pil_image.append(to_pil(cropped_image))
        except Exception as e:
            print('Parsinng error content icon', e)
            continue

    model: AutoModelForCausalLM = caption_model_processor['model']
    processor: AutoProcessor = caption_model_processor['processor']
    prompt = "<CAPTION>"
    
    generated_texts = []
    device = model.device
    for i in range(0, len(croped_pil_image), batch_size):
        batch = croped_pil_image[i:i+batch_size]
        if model.device.type == 'cuda':
            inputs = processor(images=batch, text=[prompt]*len(batch), return_tensors="pt", do_resize=False).to(device=device, dtype=torch.float16)
        else:
            inputs = processor(images=batch, text=[prompt]*len(batch), return_tensors="pt").to(device=device)
        generated_ids = model.generate(input_ids=inputs["input_ids"],pixel_values=inputs["pixel_values"],max_new_tokens=20,num_beams=1, do_sample=False)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        generated_text = [gen.strip() for gen in generated_text]
        generated_texts.extend(generated_text)
    
    return generated_texts

def get_parsed_content_icon_phi3v(filtered_boxes, ocr_bbox, image_source, caption_model_processor):
    to_pil = ToPILImage()
    if ocr_bbox:
        non_ocr_boxes = filtered_boxes[len(ocr_bbox):]
    else:
        non_ocr_boxes = filtered_boxes
    croped_pil_image = []
    for i, coord in enumerate(non_ocr_boxes):
        xmin, xmax = int(coord[0]*image_source.shape[1]), int(coord[2]*image_source.shape[1])
        ymin, ymax = int(coord[1]*image_source.shape[0]), int(coord[3]*image_source.shape[0])
        cropped_image = image_source[ymin:ymax, xmin:xmax, :]
        croped_pil_image.append(to_pil(cropped_image))

    model, processor = caption_model_processor['model'], caption_model_processor['processor']
    device = model.device
    messages = [{"role": "user", "content": "<|image_1|>\ndescribe the icon in one sentence"}] 
    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    batch_size = 5  # Number of samples per batch
    generated_texts = []

    for i in range(0, len(croped_pil_image), batch_size):
        images = croped_pil_image[i:i+batch_size]
        image_inputs = [processor.image_processor(x, return_tensors="pt") for x in images]
        inputs ={'input_ids': [], 'attention_mask': [], 'pixel_values': [], 'image_sizes': []}
        texts = [prompt] * len(images)
        for i, txt in enumerate(texts):
            input = processor._convert_images_texts_to_inputs(image_inputs[i], txt, return_tensors="pt")
            inputs['input_ids'].append(input['input_ids'])
            inputs['attention_mask'].append(input['attention_mask'])
            inputs['pixel_values'].append(input['pixel_values'])
            inputs['image_sizes'].append(input['image_sizes'])
        max_len = max([x.shape[1] for x in inputs['input_ids']])
        for i, v in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = torch.cat([processor.tokenizer.pad_token_id * torch.ones(1, max_len - v.shape[1], dtype=torch.long), v], dim=1)
            inputs['attention_mask'][i] = torch.cat([torch.zeros(1, max_len - v.shape[1], dtype=torch.long), inputs['attention_mask'][i]], dim=1)
        inputs_cat = {k: torch.concatenate(v).to(device) for k, v in inputs.items()}

        generation_args = { 
            "max_new_tokens": 25, 
            "temperature": 0.01, 
            "do_sample": False, 
        } 
        generate_ids = model.generate(**inputs_cat, eos_token_id=processor.tokenizer.eos_token_id, **generation_args) 
        # # remove input tokens 
        generate_ids = generate_ids[:, inputs_cat['input_ids'].shape[1]:]
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        response = [res.strip('\n').strip() for res in response]
        generated_texts.extend(response)

    return generated_texts

def filter_overlapping_boxes(boxes: List[Dict], iou_threshold: float) -> List[Dict]:
    """
    Filter out overlapping boxes based on IoU threshold.
    If two boxes overlap significantly, keep the larger one.

    Args:
        boxes: List of box dictionaries with 'bbox' and other metadata.
        iou_threshold: IoU threshold for considering boxes as overlapping.

    Returns:
        List of filtered box dictionaries.
    """
    filtered_boxes = []
    for i, box1_elem in enumerate(boxes):
        box1 = box1_elem['bbox']
        is_valid = True
        for j, box2_elem in enumerate(boxes):
            if i == j:
                continue
            box2 = box2_elem['bbox']
            if IoU(box1, box2) > iou_threshold and box_area(box1) > box_area(box2):
                is_valid = False
                break
        if is_valid:
            filtered_boxes.append(box1_elem)
    return filtered_boxes

def is_inside(box1, box2):
    intersection = intersection_area(box1, box2)
    ratio1 = intersection / box_area(box1)
    return ratio1 > 0.80

def process_ocr_content(
    icon_box: Dict,
    remaining_ocr: List[Dict]
) -> Optional[Dict]:
    """
    Process OCR boxes relative to an icon box, merging OCR content if applicable.
    Removes used OCR boxes from remaining_ocr.

    Args:
        icon_box: Icon box dictionary with 'bbox' and other metadata.
        remaining_ocr: List of OCR box dictionaries which may be modified by removing used boxes.

    Returns:
        Updated icon box dictionary or None if the box should be skipped.
    """
    box1 = icon_box['bbox']
    ocr_labels = []
    box_added = False

    # Iterate over a copy to avoid issues during removal
    for ocr_box_elem in list(remaining_ocr):
        box3 = ocr_box_elem['bbox']
        if is_inside(box3, box1):  # OCR inside icon
            ocr_labels.append(ocr_box_elem['content'])
            remaining_ocr.remove(ocr_box_elem)
        elif is_inside(box1, box3):  # Icon inside OCR
            box_added = True
            break

    if box_added:
        return None

    ocr_content = ' '.join(ocr_labels).strip() if ocr_labels else None
    return {
        'bbox': box1,
        'content': ocr_content,
        # 'source': 'box_yolo_content_ocr' if ocr_content else 'box_yolo_content_yolo'
    }

def remove_overlap(
    boxes: List[Dict],
    iou_threshold: float,
    ocr_bbox: Optional[List[Dict]] = None
) -> List[Dict]:
    """
    Remove overlapping boxes and merge OCR content with icon boxes.
    Valid YOLO boxes merged with OCR are added first, followed by remaining OCR boxes.

    Args:
        boxes: List of box dictionaries from YOLO with 'bbox' and other metadata.
        iou_threshold: IoU threshold for considering boxes as overlapping.
        ocr_bbox: Optional list of OCR box dictionaries.

    Returns:
        List of filtered box dictionaries with merged OCR content, followed by remaining OCR boxes.
    """
    valid_boxes = filter_overlapping_boxes(boxes, iou_threshold)
    remaining_ocr = list(ocr_bbox) if ocr_bbox else []
    filtered_boxes = []

    for box_elem in valid_boxes:
        if not ocr_bbox:
            filtered_boxes.append(box_elem)
            continue

        updated_box = process_ocr_content(box_elem, remaining_ocr)
        if updated_box:
            filtered_boxes.append(updated_box)

    # Add remaining OCR boxes that were not merged with any YOLO box
    filtered_boxes.extend(remaining_ocr)

    return filtered_boxes

def load_image(image_path: str) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.open(image_path).convert("RGB")
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed

def annotate(image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, phrases: List[str], text_scale: float, 
             text_padding=5, text_thickness=2, thickness=3) -> np.ndarray:
    """    
    This function annotates an image with bounding boxes and labels.

    Parameters:
    image_source (np.ndarray): The source image to be annotated.
    boxes (torch.Tensor): A tensor containing bounding box coordinates. in cxcywh format, pixel scale
    logits (torch.Tensor): A tensor containing confidence scores for each bounding box.
    phrases (List[str]): A list of labels for each bounding box.
    text_scale (float): The scale of the text to be displayed. 0.8 for mobile/web, 0.3 for desktop # 0.4 for mind2web

    Returns:
    np.ndarray: The annotated image.
    """
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    xywh = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xywh").numpy()
    detections = sv.Detections(xyxy=xyxy)

    labels = [f"{phrase}" for phrase in range(boxes.shape[0])]

    box_annotator = BoxAnnotator(text_scale=text_scale, text_padding=text_padding,text_thickness=text_thickness,thickness=thickness) # 0.8 for mobile/web, 0.3 for desktop # 0.4 for mind2web
    annotated_frame = image_source.copy()
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels, image_size=(w,h))

    label_coordinates = {f"{phrase}": v for phrase, v in zip(phrases, xywh)}
    return annotated_frame, label_coordinates

def predict_yolo(model, image, box_threshold, imgsz, scale_img, iou_threshold=0.7):
    """ Use huggingface model to replace the original model
    """
    # model = model['model']
    if scale_img:
        result = model.predict(
        source=image,
        conf=box_threshold,
        imgsz=imgsz,
        iou=iou_threshold, # default 0.7
        )
    else:
        result = model.predict(
        source=image,
        conf=box_threshold,
        iou=iou_threshold, # default 0.7
        )
    boxes = result[0].boxes.xyxy#.tolist() # in pixel space
    conf = result[0].boxes.conf
    phrases = [str(i) for i in range(len(boxes))]

    return boxes, conf, phrases

def int_box_area(box, w, h):
    x1, y1, x2, y2 = box
    int_box = [int(x1*w), int(y1*h), int(x2*w), int(y2*h)]
    area = (int_box[2] - int_box[0]) * (int_box[3] - int_box[1])
    return area

def get_som_labeled_img(
        image_source: Union[str, Image.Image], 
        model=None, 
        BOX_TRESHOLD=0.01, 
        output_coord_in_ratio=False, 
        ocr_bbox=None, 
        text_scale=0.4, 
        text_padding=5, 
        draw_bbox_config=None, 
        caption_model_processor=None, 
        ocr_text=[], 
        use_local_semantics=True, 
        iou_threshold=0.9, 
        scale_img=False, 
        imgsz=None, 
        batch_size=128):
    """Process either an image path or Image object
    
    Args:
        image_source: Either a file path (str) or PIL Image object
        ...
    """
    if isinstance(image_source, str):
        image_source = Image.open(image_source)
    image_source = image_source.convert("RGB") # for CLIP
    w, h = image_source.size
    if not imgsz:
        imgsz = (h, w)
    xyxy, logits, phrases = predict_yolo(model=model, image=image_source, box_threshold=BOX_TRESHOLD, imgsz=imgsz, scale_img=scale_img, iou_threshold=0.1)
    xyxy = xyxy / torch.Tensor([w, h, w, h]).to(xyxy.device)
    image_source = np.asarray(image_source)
    phrases = [str(i) for i in range(len(phrases))]

    if ocr_bbox:
        ocr_bbox = torch.tensor(ocr_bbox) / torch.Tensor([w, h, w, h])
        ocr_bbox=ocr_bbox.tolist()
    else:
        print('no ocr bbox!!!')
        ocr_bbox = None

    ocr_bbox_elem = [{'bbox':box, 'content':txt} for box, txt in zip(ocr_bbox, ocr_text) if int_box_area(box, w, h) > 0] 
    xyxy_elem = [{'bbox':box, 'content':None} for box in xyxy.tolist() if int_box_area(box, w, h) > 0]
    filtered_boxes = remove_overlap(boxes=xyxy_elem, iou_threshold=iou_threshold, ocr_bbox=ocr_bbox_elem)
    
    # sort the filtered_boxes so that the one with 'content': None is at the end, and get the index of the first 'content': None
    filtered_boxes_elem = sorted(filtered_boxes, key=lambda x: x['content'] is None)
    # get the index of the first 'content': None
    starting_idx = next((i for i, box in enumerate(filtered_boxes_elem) if box['content'] is None), -1)
    filtered_boxes = torch.tensor([box['bbox'] for box in filtered_boxes_elem])

    if use_local_semantics:
        caption_model = caption_model_processor['model']
        if 'phi3_v' in caption_model.config.model_type: 
            parsed_content_icon = get_parsed_content_icon_phi3v(filtered_boxes, ocr_bbox, image_source, caption_model_processor)
        else:
            parsed_content_icon = get_parsed_content_icon(filtered_boxes, starting_idx, image_source, caption_model_processor, batch_size=batch_size)
        for i, box in enumerate(filtered_boxes_elem):
            if box['content'] is None:
                box['content'] = parsed_content_icon.pop(0)

    filtered_boxes = box_convert(boxes=filtered_boxes, in_fmt="xyxy", out_fmt="cxcywh")
    phrases = [i for i in range(len(filtered_boxes))]
    
    # draw boxes
    if draw_bbox_config:
        annotated_frame, label_coordinates = annotate(image_source=image_source, boxes=filtered_boxes, logits=logits, phrases=phrases, **draw_bbox_config)
    else:
        annotated_frame, label_coordinates = annotate(image_source=image_source, boxes=filtered_boxes, logits=logits, phrases=phrases, text_scale=text_scale, text_padding=text_padding)
    
    pil_img = Image.fromarray(annotated_frame)
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode('ascii')
    if output_coord_in_ratio:
        label_coordinates = {k: [v[0]/w, v[1]/h, v[2]/w, v[3]/h] for k, v in label_coordinates.items()}
        assert w == annotated_frame.shape[1] and h == annotated_frame.shape[0]
    for key in label_coordinates:
        label_coordinates[key] = np.round(label_coordinates[key]).astype(int).tolist()
    filtered_boxes_elem = [{key: value for key, value in el.items() if not key.startswith('bb')} for el in filtered_boxes_elem]
    for i, elem in enumerate(filtered_boxes_elem):
        elem['coordinates'] = label_coordinates[str(i)][:2]  # Используем get для безопасности
    return encoded_image, filtered_boxes_elem

def get_xywh(input):
    x, y, w, h = input[0][0], input[0][1], input[2][0] - input[0][0], input[2][1] - input[0][1]
    x, y, w, h = int(x), int(y), int(w), int(h)
    return x, y, w, h

def get_xyxy(input):
    x, y, xp, yp = input[0][0], input[0][1], input[2][0], input[2][1]
    x, y, xp, yp = int(x), int(y), int(xp), int(yp)
    return x, y, xp, yp

def check_ocr_box(image_source: Union[str, Image.Image], display_img = True, output_bb_format='xywh', goal_filtering=None, easyocr_args=None):
    if isinstance(image_source, str):
        image_source = Image.open(image_source)
    if image_source.mode == 'RGBA':
        # Convert RGBA to RGB to avoid alpha channel issues
        image_source = image_source.convert('RGB')
    image_np = np.array(image_source)
    if easyocr_args is None:
            easyocr_args = {}
    result = reader.readtext(image_np, **easyocr_args)
    coord = [item[0] for item in result]
    text = [item[1] for item in result]
    if display_img:
        opencv_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        bb = []
        for item in coord:
            x, y, a, b = get_xywh(item)
            bb.append((x, y, a, b))
            cv2.rectangle(opencv_img, (x, y), (x+a, y+b), (0, 255, 0), 2)
        #  matplotlib expects RGB
        plt.imshow(cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB))
    else:
        if output_bb_format == 'xywh':
            bb = [get_xywh(item) for item in coord]
        elif output_bb_format == 'xyxy':
            bb = [get_xyxy(item) for item in coord]
    return (text, bb), goal_filtering