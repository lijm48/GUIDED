from tqdm import tqdm

import torch
from torch import IntTensor, Tensor
from torchvision.ops import batched_nms

import pickle, json
import argparse
import os

DEVICE = 'cpu'

def save_object(obj, path):
    """"Save an object using the pickle library on a file
    
    :param obj: undefined. Object to save
    :param fileName: str. Name of the file of the object to save
    """
    print("Saving " + path)
    with open(path, 'wb') as fid:
        pickle.dump(obj, fid)
        
def load_object(path):
    """"Load an object from a file
    
    :param fileName: str. Name of the file of the object to load
    :return: obj: undefined. Object loaded
    """
    try:
        with open(path, 'rb') as fid:
            obj = pickle.load(fid)
            return obj
    except IOError:
        return None   

def read_json(file_name):
    #Read JSON file
    with open(file_name) as infile:
        data = json.load(infile)
    return data

def write_json(data, file_name):
    # Write JSON file
    with open(file_name, "w") as outfile:
        json.dump(data, outfile)

# DATA HANDLING UTILITIES

def transform_predslist_to_dict(preds):
    result = {}
    for pred in preds:
        image = pred['image_filepath']
        if image not in result:
            result[image] = []
        result[image].append(pred)
    return result  

def get_image_ground_truth(data, image_id):
    """
    Given a dictionary 'data' and an 'image_id', returns a dictionary with 'boxes' and 'categories' information for
    that image.

    Args:
        data (dict): The data dictionary containing 'annotations'.
        image_id (int): The image_id for which to retrieve data.

    Returns:
        dict: A dictionary with 'boxes' and 'categories' information for the given image_id.
    """
    image_data = {'boxes': [], 'labels': [], 'annotation_id': []}  # Initialize the dictionary to store image data

    # Loop through each annotation in the 'annotations' list
    for annotation in data['annotations']:
        if annotation['image_id'] == image_id:
            # If the 'image_id' in the annotation matches the given 'image_id', append bbox and category_id to the lists
            image_data['annotation_id'].append(annotation['id']) 
            image_data['boxes'].append(annotation['bbox'])
            image_data['labels'].append(annotation['category_id'])

    image_data['boxes'] = convert_format(image_data['boxes'])
    # tensorize elements
    image_data['boxes'] = Tensor(image_data['boxes']).to(DEVICE)
    image_data['labels'] = IntTensor(image_data['labels']).to(DEVICE)
    
    return image_data

def calculate_iou(box1, box2):
    # Extract coordinates
    x1_box1, y1_box1, x2_box1, y2_box1 = box1
    x1_box2, y1_box2, x2_box2, y2_box2 = box2
    
    # Calculate intersection coordinates
    x1_inter = max(x1_box1, x1_box2)
    y1_inter = max(y1_box1, y1_box2)
    x2_inter = min(x2_box1, x2_box2)
    y2_inter = min(y2_box1, y2_box2)
    
    # Calculate area of intersection
    intersection_area = max(0, x2_inter - x1_inter + 1) * max(0, y2_inter - y1_inter + 1)
    
    # Calculate area of both bounding boxes
    area_box1 = (x2_box1 - x1_box1 + 1) * (y2_box1 - y1_box1 + 1)
    area_box2 = (x2_box2 - x1_box2 + 1) * (y2_box2 - y1_box2 + 1)
    
    # Calculate Union
    union_area = area_box1 + area_box2 - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area
    
    return iou

def convert_format(boxes):
    for box in boxes:
        box[2] += box[0]
        box[3] += box[1]
    return boxes

def get_image_preds(preds, include_total_scores=True):
    labels = []
    scores = []
    boxes = []
    total_scores = []
    for pred in preds:
        labels += [x for x in pred['labels']]
        scores += [x for x in pred['scores']]
        boxes += ([x for x in pred['boxes']])
        if include_total_scores:
            total_scores += [x for x in pred['total_scores']]
        
    if type(boxes[0]) != torch.Tensor:
        return {
            'boxes': Tensor(boxes).to(DEVICE),
            'labels': IntTensor(labels).to(DEVICE),
            'scores': Tensor(scores).to(DEVICE),
            'total_scores': Tensor(total_scores).to(DEVICE),
            'category_id': preds[0]['category_id']
        }
    else:
        return {
            'boxes': torch.stack(boxes, dim=0).to(DEVICE),
            'labels': IntTensor(labels).to(DEVICE),
            'scores': Tensor(scores).to(DEVICE),
            'total_scores': Tensor(total_scores).to(DEVICE),
            'category_id': preds[0]['category_id']
        }

class CustomMetrics():
    intersected_predictions = []
    position_array = []
    
    def __init__(self) -> None:
        intersected_predictions = []
        position_array = []
        pass
    
    def update(self, targets, preds, nms=True, iou=0.5, verbose=False, one_inference_at_time=False, n_neg=10):
        """
        target: Detection dataset in standard COCO format
        preds: List of detections from a dataset where each one has the following fields: 'scores', 'boxes' (in the format xyxy), 'labels', 'total_socres', 'image_id'
        """
        
        targets['annotations'] = [ann for ann in targets['annotations'] if len(ann['neg_category_ids']) >= n_neg]
        
        if one_inference_at_time:
            return self.__update_one_inference_at_time__(targets, preds, iou=iou, verbose=verbose)
        
        self.intersected_predictions = []
        self.position_array = []
        
        # transforming list to dict
        preds = transform_predslist_to_dict(preds)
        
        # initializing counters
        n_images = 0
        count_no_intersections = 0
        
        # iterating over images
        if verbose:
            targets['images'] = tqdm(targets['images'])
        for imm in targets['images']:
            target = get_image_ground_truth(targets, imm['id'])
            if imm['file_name'] in preds:
                imm_preds = [get_image_preds([pred_per_cat]) for pred_per_cat in preds[imm['file_name']]]
            else:
                continue
            n_images += 1
            # iterating over the predictions per category
            for imm_preds_per_cat in imm_preds:
                # appliyng NMS to preds if nms is True
                if nms:
                    to_keep = batched_nms(imm_preds_per_cat['boxes'], 
                                        imm_preds_per_cat['scores'], 
                                        torch.IntTensor([0] * len(imm_preds_per_cat['boxes'])),
                                        iou)
                    imm_preds_per_cat['boxes'], imm_preds_per_cat['scores'], imm_preds_per_cat['labels'], imm_preds_per_cat['total_scores'] = imm_preds_per_cat['boxes'][to_keep], \
                                                                                                            imm_preds_per_cat['scores'][to_keep], \
                                                                                                            imm_preds_per_cat['labels'][to_keep], \
                                                                                                            imm_preds_per_cat['total_scores'][to_keep]
                # iterating over targets
                for box, label, id in zip(target['boxes'], target['labels'], target['annotation_id']):
                    if label != imm_preds_per_cat['category_id']:
                        continue
                    # applying NMS class agnostics to the preds concatenated with target, with targets score = 1
                    to_remove = batched_nms(torch.cat((imm_preds_per_cat['boxes'], box.unsqueeze(0)), dim=0), 
                                            torch.cat((imm_preds_per_cat['scores'], IntTensor([1])), dim=0), 
                                            torch.IntTensor([0] * (len(imm_preds_per_cat['boxes']) + 1)),
                                            iou)
                    # check which is the deleted box with higher confidences
                    deleted_elements = sorted(list(set(range(len(imm_preds_per_cat['scores']))) - set(to_remove.tolist()[1:])))
                    if len(deleted_elements) > 0:
                        # iou of deleted element needs to be over IoU with the GT
                        assert calculate_iou(imm_preds_per_cat['boxes'][deleted_elements[0]].tolist(), box) >= iou, "iou of deleted element needs to be over IoU with the GT"
                        
                        # appending to the list of predicted confidence the element
                        self.intersected_predictions.append({
                            'annotation_id': id,
                            'total_scores': imm_preds_per_cat['total_scores'][deleted_elements[0]].tolist()
                        })
                    else:
                        count_no_intersections += 1
                        self.intersected_predictions.append({
                            'annotation_id': id,
                            'total_scores': imm_preds_per_cat['total_scores'].shape[1] * [0]
                        })
                    # we store the rank of the prediction in the position array congruently
                    # we see the rank as the number of confidence in the list with higher or equal value - 1
                    last_pred = self.intersected_predictions[-1]
                    rank = sum(1 for conf in last_pred['total_scores'] if conf >= last_pred['total_scores'][0]) - 1
                    self.position_array.append(rank + 1)
        
        
        assert len(self.intersected_predictions) == len(self.position_array), "Incongruent list dimensions"
        if verbose:
            print("Number of images: %d" % n_images)
            print("Number of no intersection with GT: %d" % count_no_intersections)
        return self.intersected_predictions, self.position_array
    
    def get_median_rank(self):
        return sorted(self.position_array)[(len(self.position_array) // 2) + 1]
            
    def get_medium_rank(self):
        return sum(elem for elem in self.position_array) / len(self.position_array)
    
    # PRIVATE METHODS
    def __update_one_inference_at_time__(self, targets, preds, nms=True, iou=0.5, verbose=False, one_inference_at_time=False):
        """
        target: Detection dataset in standard COCO format
        preds: List of detections from a dataset where each one has the following fields: 'scores', 'boxes' (in the format xyxy), 'labels', 'total_socres', 'image_id'
        """
        
        self.intersected_predictions = []
        self.position_array = []
        
        # transforming list to dict
        preds = transform_predslist_to_dict(preds)
        
        # initializing counters
        n_images = 0
        count_no_intersections = 0
        
        def assign_max_scores(scores, ious, labels, iou_thresh):
            label_offset = int(min(labels))
            total_scores = [0] * (int(max(labels) - min(labels)) + 1)

            for score, iou, label in zip(scores, ious, labels):
                if iou < iou_thresh:
                    continue
                if total_scores[int(label - label_offset)] < score:
                    total_scores[int(label - label_offset)] = float(score)
            
            return total_scores

        # iterating over images
        if verbose:
            targets['images'] = tqdm(targets['images'])
        for imm in targets['images']:
            target = get_image_ground_truth(targets, imm['id'])
            if imm['file_name'] in preds:
                imm_preds = [get_image_preds([pred_per_cat], include_total_scores=False) for pred_per_cat in preds[imm['file_name']]]
            else:
                continue
            n_images += 1
            # iterating over the predictions per category
            for imm_preds_per_cat in imm_preds:
                for box, label, id in zip(target['boxes'], target['labels'], target['annotation_id']):
                    ious = [float(calculate_iou(box, pred_box)) for pred_box in imm_preds_per_cat['boxes']]
                    total_scores = assign_max_scores(imm_preds_per_cat['scores'], ious, imm_preds_per_cat['labels'], iou)
                    self.intersected_predictions.append({
                        'annotation_id': id,
                        'total_scores': total_scores
                    })
                    # we store the rank of the prediction in the position array congruently
                    # we see the rank as the number of confidence in the list with higher or equal value - 1
                    last_pred = self.intersected_predictions[-1]
                    rank = sum(1 for conf in last_pred['total_scores'] if conf >= last_pred['total_scores'][0]) - 1
                    self.position_array.append(rank + 1)
                
        return self.intersected_predictions, self.position_array

def clip_preds(data, n):
    
    for pred_group in data:
        new_vocabulary = pred_group['vocabulary'][:n + 1]
        keep = [i for i in range(len(pred_group['labels'])) if pred_group['labels'][i] in new_vocabulary][:50]
        pred_group['boxes'] = [pred_group['boxes'][i] for i in keep]
        pred_group['labels'] = [pred_group['labels'][i] for i in keep]
        pred_group['scores'] = [pred_group['scores'][i] for i in keep]
        
    return data
      
def main(args: argparse.Namespace):
    data = read_json(args.ground_truth)
    print(f"Defining CustomMetrics for {os.path.basename(args.ground_truth)}...")
    print(f"evaluating {os.path.basename(args.predictions)}...")
    custom_metrics = CustomMetrics()
    preds = load_object(args.predictions)
    custom_metrics.update(data, preds, verbose=True, one_inference_at_time=True)
    print("Medium: %s" % custom_metrics.get_medium_rank())
    print("Median: %s" % custom_metrics.get_median_rank())
    # custom_metrics = CustomMetrics()
    # preds = load_object("2clip_gdino.pkl")
    # custom_metrics.update(data, preds, verbose=True, one_inference_at_time=True)
    # print("Medium: %s" % custom_metrics.get_medium_rank())
    # print("Median: %s" % custom_metrics.get_median_rank())
    
    
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--ground_truth", type=str, default="1_attributes")
    args.add_argument("--predictions", type=str, default="1_attributes")
    
    args = args.parse_args()
    main(args)
            
        
