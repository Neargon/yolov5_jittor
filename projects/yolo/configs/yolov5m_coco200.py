_base_ = ['yolo_model_base.py', 'yolo_dataset_base.py', 'yolo_optimizer_base.py', 'yolo_scheduler_base.py']
batch_size = 1
max_epoch = 300
log_interval=1
eval_interval=10
checkpoint_interval = 10
stride=32
imgsz=640
imgsz_test=640
dataset_type = 'YoloDataset'

model = dict(
    type='YOLOv5M',
    ema=True,
    imgsz=imgsz
)
parameter_groups_generator = dict(
    batch_size=batch_size
)
scheduler=dict(
    max_steps=max_epoch
)
dataset = dict(
    val=dict(
        path="C:\\Users\\Admin\\Desktop\\yolox\\dataset\\labels\\val.txt",
        batch_size = batch_size,
        num_workers=8,
        stride=stride,
        imgsz=imgsz_test

        ),
    test=dict(
        path='C:\\Users\\Admin\\Desktop\\yolox\\dataset\\labels\\test.txt',
        batch_size = batch_size,
        num_workers=8,
        stride=stride,
        imgsz=imgsz_test
        ),
    train=dict(
        path='C:\\Users\\Admin\\Desktop\\yolox\\dataset\\labels\\train.txt',
        batch_size = batch_size,
        num_workers=8,
        stride=stride,
        imgsz=imgsz,
        augment=True
        )
)

logger = dict(
    type="RunLogger")