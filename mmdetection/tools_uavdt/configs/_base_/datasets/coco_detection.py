dataset_type = 'UAVDTDataset'
data_root = '/home/twsf/data/UAVDT/density_chip/'
img_norm_cfg = dict(
    mean=[93.77, 96.338, 94.527], std=[43.93, 41.47, 43.667], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        # /home/twsf/data/Visdrone/VisDrone2019-DET-val/annotations_json/
        ann_file=data_root + "Annotations_json/instances_train.json",
        img_prefix=data_root + 'JPEGImages',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + "Annotations_json/instances_val.json",
        img_prefix=data_root + 'JPEGImages',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
