dataset_type = 'VisdroneDataset'
data_root = '/home/twsf/data/Visdrone/density_chip/'
img_norm_cfg = dict(
    mean=[95.115, 96.39, 93.075], std=[48.96, 46.665, 49.47], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=[(800, 800), (1024, 1024)], keep_ratio=True),
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
        img_scale=(1024, 1024),
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
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        # /home/twsf/data/Visdrone/VisDrone2019-DET-val/annotations_json/
        ann_file=data_root + "Annotations_json/instances_traintest.json",
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
