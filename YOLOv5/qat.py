import os
import time
import sys
import argparse
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision
from torchvision.io import read_image
from pytorch_nndct.apis import torch_quantizer
from models.common import DetectMultiBackend
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, label_dir, img_dir, width, height, transforms=None):
        self.label_dir = label_dir
        self.img_dir = img_dir
        self.transforms = transforms
        self.height = height
        self.width = width

        self.img_names = []
        for filename in os.listdir(img_dir):
            if not filename.startswith('.ipynb_checkpoints'):  # Exclude .ipynb_checkpoints file
                temp = os.path.splitext(filename)
                self.img_names.append(temp[0])

    def gen_id(name: str):
        name = ''.join((x for x in name if x.isdigit()))
        name = name[0:10] + name[11:len(name)]
        return int(name)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_filename = self.img_names[idx] + ".png"
        label_filename = self.img_names[idx] + ".txt"
        img_path = os.path.join(self.img_dir, img_filename)
        label_path = os.path.join(self.label_dir, label_filename)

        # Read the image using PIL
        image_pil = Image.open(img_path)

        # Convert RGBA to RGB if it's RGBA
        if image_pil.mode == 'RGBA':
            image_pil = image_pil.convert('RGB')

        # Convert PIL Image to a PyTorch tensor
        image = torchvision.transforms.functional.to_tensor(image_pil)

        # The rest of the code remains the same
        image = torchvision.transforms.Resize((self.width, self.height))(image)
        image = image.float()
        image /= 255

        boxes_array = []
        labels_array = []

        with open(label_path) as f:
            lines = f.readlines()
            for line in lines:
                vals = line.split(" ")
                labels_array.append(int(vals[0]))
                x0 = (float(vals[1]) - (float(vals[3]) / 2)) * self.width
                y0 = (float(vals[2]) - (float(vals[4]) / 2)) * self.height
                x1 = (float(vals[1]) + (float(vals[3]) / 2)) * self.width
                y1 = (float(vals[2]) + (float(vals[4]) / 2)) * self.height
                boxes_array.append([x0, y0, x1, y1])

        boxes_tensor = torch.as_tensor(boxes_array, dtype=torch.float32)
        area_tensor = (boxes_tensor[:, 3] - boxes_tensor[:, 1]) * (boxes_tensor[:, 2] - boxes_tensor[:, 0])
        iscrowd_tensor = torch.zeros((boxes_tensor.shape[0],), dtype=torch.int64)
        labels_tensor = torch.as_tensor(labels_array, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes_tensor
        target["labels"] = labels_tensor
        target["area"] = area_tensor
        target["iscrowd"] = iscrowd_tensor
        # img_id = torch.tensor([self.gen_id(self.img_names[idx])])
        img_id = torch.tensor([idx + 1])
        target["image_id"] = img_id

        if self.transforms:
            sample = self.transform(image=image, bboxes=target["boxes"], labels=labels_tensor)
            image = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])

        return image, target

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def non_max_suppression(
        prediction,
        conf_thres=0.45,
        iou_thres=0.25,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0,  # number of masks
):
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]

    device = prediction.device
    mps = 'mps' in device.type
    if mps:
        prediction = prediction.cpu()
    bs = prediction.shape[0]
    nc = 1
    xc = prediction[..., 4] > conf_thres

    max_wh = 7680
    max_nms = 30000
    time_limit = 0.5 + 0.05 * bs
    redundant = True
    multi_label &= nc > 1
    merge = False

    t = time.time()
    mi = 5 + nc
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]

        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]
            v[:, 4] = 1.0
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0
            x = torch.cat((x, v), 0)

        if not x.shape[0]:
            continue

        x[:, 5:] *= x[:, 4:5]
        box = xywh2xyxy(x[:, :4])
        mask = x[:, mi:]

        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]
        if not n:
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        c = x[:, 5:6] * max_wh if not agnostic else torch.zeros_like(x[:, 5:6])  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores

        i = torchvision.ops.nms(boxes, scores, iou_thres)
        i = i[:max_det]
        if merge and (1 < n < 3E3):
            iou = box_iou(boxes[i], boxes) > iou_thres
            weights = iou * scores[None]
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)
            if redundant:
                i = i[iou.sum(1) > 1]

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            LOGGER.warning(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            break

    return output


DIVIDER = '-'*50

def quantize(build_dir, quant_mode, weights, dataset, print_detections=True):
    quant_model = build_dir + '/quant_model'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DetectMultiBackend(weights=weights)

    rand_in = torch.randn(1, 3, 640, 640)
    quantizer = torch_quantizer(quant_mode, model, rand_in, output_dir=quant_model)
    quantized_model = quantizer.quant_model
    quantized_model = quantized_model.to(device)

    train_dataset = CustomImageDataset(os.path.join(dataset + 'labels'), os.path.join(dataset + 'images'), 640, 640)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
    
    start_time = time.time()
    quantized_model.eval()

    with torch.no_grad():
        for image, target in train_loader:
            print(f'Image {target["image_id"][0][0]}')
            output = quantized_model(image.to(device))
            pred = non_max_suppression(output)

            if print_detections and quant_mode == 'test':
                for i, detections in enumerate(pred):
                    print(f"Image {target['image_id'][i][0]} Detections:")
                    for det in detections:
                        confidence = det[4]*0.005
                        if confidence >= 0.1:
                            print(f"  Co-ordinates: {det[:4]}, Confidence: {confidence}, Class Index: {det[5]}")
                            
        if quant_mode == 'test':
            print(f" mAP@50-90: {det[6]}")
            end_time = time.time()
            execution_time = (end_time - start_time)/2
            print(f"Inference time: {execution_time:.2f} ms")


    if quant_mode == 'calib':
        quantizer.export_quant_config()
    if quant_mode == 'test':
        quantizer.export_torch_script()
        quantizer.export_onnx_model()
        quantizer.export_xmodel(deploy_check=False, output_dir=quant_model)
  

def run_main():

  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument('-b',  '--build_dir',  type=str, default='build',    help='Path to build folder. Default is build')
  ap.add_argument('-q',  '--quant_mode', type=str, default='calib',    choices=['calib','test'], help='Quantization mode (calib or test). Default is calib')
  ap.add_argument('-w',  '--weights',  type=str,  help='Path to yolo weights file')
  ap.add_argument('-d',  '--dataset',  type=str,  help='Path to your calibration directory with subdirectories called "images" and "labels"' )
  args = ap.parse_args()

  print('\n'+DIVIDER)
  print('PyTorch version : ',torch.__version__)
  print(sys.version)
  print(DIVIDER)
  print(' Command line options:')
  print ('--build_dir    : ',args.build_dir)
  print ('--quant_mode   : ',args.quant_mode)
  print ('--weights    : ',args.weights)
  print ('--dataset    : ',args.dataset)
  print(DIVIDER)

  quantize(args.build_dir, args.quant_mode, args.weights, args.dataset, print_detections=True)
  return

if __name__ == '__main__':
    run_main()