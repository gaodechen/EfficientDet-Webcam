## EfficientDet on Multiple Webcams

* **EfficientDet Implementation**: [zylo117/Yet-Another-EfficientDet-Pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)

* **Multiple Webcams with Object Detection**: [gaodechen/webcam_yolov3_jetson_tx_hikvision](https://github.com/gaodechen/webcam_yolov3_jetson_tx_hikvision)

## Usage

Same usage as [gaodechen/webcam_yolov3_jetson_tx_hikvision](https://github.com/gaodechen/webcam_yolov3_jetson_tx_hikvision).

## Multiple Webcams Streaming & Object Detection

- `model.py`: post-packaging for `Yet-Another-EfficientDet-Pytorch`, no need to change
- `settings.py`: IP list & image shape could be modified here, view the  [zylo117/Yet-Another-EfficientDet-Pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) repo for other configurations

Running:

```
python run.py --single_window=True (or False) --num_cameras=4
```

`single_window`: argument used when multiple images should be merged and displayed into one single window.

`num_cameras`: number of cameras to be processed.

### Streaming Only

Delete model processing part in predict() as below.

```python
def predict(raw_q, pred_q):
    # model = Model()
    while True:
        raw_img = raw_q.get()
        # pred_img = model.run(raw_img)
        pred_q.put(raw_img)
```
