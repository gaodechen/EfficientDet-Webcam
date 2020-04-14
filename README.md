## EfficientDet on Multiple Webcams

* **EfficientDet Implementation**: [zylo117/Yet-Another-EfficientDet-Pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)

* **Multiple Webcams with Object Detection**: [gaodechen/webcam_yolov3_jetson_tx_hikvision](https://github.com/gaodechen/webcam_yolov3_jetson_tx_hikvision)

## Usage

Same usage as [gaodechen/webcam_yolov3_jetson_tx_hikvision](https://github.com/gaodechen/webcam_yolov3_jetson_tx_hikvision).

## 多网络摄像头拉流 + EfficientDet对象检测

- `model.py`是对`Yet-Another-EfficientDeg-Pytorch`模型推断的二次封装
- `settings.py`当中修改IP camera地址列表，画面大小

运行：

```
python run.py --single_window=True (or False) --num_cameras=4
```

使用`single_window`则所有画面合并显示到同一个窗口当中，`num_cameras`是使用IP camera个数。

### 仅多摄像头拉流

去掉`predict()`进程的调用，并且`pop_image()`显示`raw_q`中的原图像。

```python
processes = [
    mp.Process(target=push_image, args=(raw_q, cam_addr)),
    # display images in raw_q instead
    mp.Process(target=pop_image, args=(raw_q, window_name)),
]
```
