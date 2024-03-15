#pretrained = 'https://download.openmmlab.com/mmclassification/v0/cspnet/cspdarknet50_3rdparty_8xb32_in1k_20220329-bd275287.pth'


# model settings
model = dict(
    type='FasterRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(  # 主干网络的配置文件
        type='ResNet',  # 主干网络的类别，可用选项请参考 https://mmdetection.readthedocs.io/en/latest/api.html#mmdet.models.backbones.ResNet
        depth=50,  # 主干网络的深度，对于 ResNet 和 ResNext 通常设置为 50 或 101
        num_stages=4,  # 主干网络状态(stages)的数目，这些状态产生的特征图作为后续的 head 的输入
        out_indices=(0, 1, 2, 3),  # 每个状态产生的特征图输出的索引
        frozen_stages=1,  # 第一个状态的权重被冻结
        norm_cfg=dict(  # 归一化层(norm layer)的配置项
            type='BN',  # 归一化层的类别，通常是 BN 或 GN
            requires_grad=True),  # 是否训练归一化里的 gamma 和 beta
        norm_eval=True,  # 是否冻结 BN 里的统计项
        style='pytorch',  # 主干网络的风格，'pytorch' 意思是步长为2的层为 3x3 卷积， 'caffe' 意思是步长为2的层为 1x1 卷积
        #init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),  # 加载通过 ImageNet 预训练的模型,
    ),
    neck=dict(
        type='NASFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        stack_times=5,
        start_level=1,
        norm_cfg=dict(type='BN', requires_grad=True)),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss', use_sigmoid=True, loss_weight=1.0), #CrossEntropyLoss
        loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0)), #L1Loss
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(
                type='ModulatedDeformRoIPoolPack', 
                output_size=7,
                sampling_ratio=0,
                output_channels=256), #RoIAlign
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='FocalLoss', use_sigmoid=True, loss_weight=1.0), #CrossEntropyLoss
            loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0))), #L1Loss
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='PseudoSampler', #RandomSampler
                ),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='PseudoSampler'),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))
"""

dataset settings

"""


# 修改数据集相关配置
"""metainfo = {
    'classes': ('banana', ),
    'palette': [
        (220, 20, 60),
    ]
}"""
"""train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='bananas_train/coco_train.json',
        data_prefix=dict(img='bananas_train/images/')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='bananas_train/coco_valid.json',
        data_prefix=dict(img='bananas_train/images/')))
test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='bananas_test/coco_test.json',
        data_prefix=dict(img='bananas_test/images/')))
"""

dataset_type = 'CocoDataset'  # 数据集类型，这将被用来定义数据集。
data_root = 'data/coco/'  # 数据集的根目录
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'instances'))
]
train_dataloader = dict(  # 训练 dataloader 配置
    batch_size=16,  # 单个 GPU 的 batch size
    num_workers=14,  # 单个 GPU 分配的数据加载线程数
    persistent_workers=True,  # 如果设置为 True，dataloader 在迭代完一轮之后不会关闭数据读取的子进程，可以加速训练
    sampler=dict(  # 训练数据的采样器
        type='DefaultSampler',  # 默认的采样器，同时支持分布式和非分布式训练。请参考 https://mmengine.readthedocs.io/zh_CN/latest/api/generated/mmengine.dataset.DefaultSampler.html#mmengine.dataset.DefaultSampler
        shuffle=True),  # 随机打乱每个轮次训练数据的顺序
    batch_sampler=dict(type='AspectRatioBatchSampler'),  # 批数据采样器，用于确保每一批次内的数据拥有相似的长宽比，可用于节省显存
    dataset=dict(  # 训练数据集的配置
        type=dataset_type,
        data_root=data_root,
        ann_file='instances_train2017.1@1.json',  # annotations/instances_train2017.json
        data_prefix=dict(img='train2017/'),  # 图片路径前缀
        filter_cfg=dict(filter_empty_gt=True, min_size=32),  # 图片和标注的过滤配置
        pipeline=train_pipeline))  # 这是由之前创建的 train_pipeline 定义的数据处理流程。
val_dataloader = dict(  # 验证 dataloader 配置
    batch_size=1,  # 单个 GPU 的 Batch size。如果 batch-szie > 1，组成 batch 时的额外填充会影响模型推理精度
    num_workers=14,  # 单个 GPU 分配的数据加载线程数
    persistent_workers=True,  # 如果设置为 True，dataloader 在迭代完一轮之后不会关闭数据读取的子进程，可以加速训练
    drop_last=False,  # 是否丢弃最后未能组成一个批次的数据
    sampler=dict(
        type='DefaultSampler',
        shuffle=False),  # 验证和测试时不打乱数据顺序
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,  # 开启测试模式，避免数据集过滤图片和标注
        pipeline=test_pipeline))
test_dataloader = val_dataloader  # 测试 dataloader 配置


val_evaluator = dict(  # 验证过程使用的评测器
    type='CocoMetric',  # 用于评估检测和实例分割的 AR、AP 和 mAP 的 coco 评价指标
    ann_file=data_root + 'annotations/instances_val2017.json',  # 标注文件路径
    metric=['bbox'],#'segm'  # 需要计算的评价指标，`bbox` 用于检测，`segm` 用于实例分割
    format_only=False)
test_evaluator = val_evaluator  # 测试过程使用的评测器

optim_wrapper = dict(  # 优化器封装的配置
    type='OptimWrapper',  # 优化器封装的类型。可以切换至 AmpOptimWrapper 来启用混合精度训练
    optimizer=dict(  # 优化器配置。支持 PyTorch 的各种优化器。请参考 https://pytorch.org/docs/stable/optim.html#algorithms
        type='Adam',  # 随机梯度下降优化器
        lr=0.02,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.0001), # 权重衰减
    clip_grad=None,  # 梯度裁剪的配置，设置为 None 关闭梯度裁剪。使用方法请见 https://mmengine.readthedocs.io/en/latest/tutorials/optimizer.html
    )

# iter-based 训练配置
train_cfg = dict(
    type='EpochBasedTrainLoop',  # iter-based 训练循环
    max_epochs=10,  # 最大迭代次数
    val_interval=1)  # 每隔多少次进行一次验证

val_cfg = dict(type='ValLoop')  # 验证循环的类型
test_cfg = dict(type='TestLoop')  # 测试循环的类型

param_scheduler = [
    # 在 [0, 2) 迭代时使用线性学习率
    dict(type='LinearLR',
         start_factor=0.001,
         by_epoch=True,
         begin=0,
         end=2),
    # 在 [2, 10) 迭代时使用余弦学习率
    dict(type='CosineAnnealingLR',
         T_max=8,
         by_epoch= True,
         begin=2,
         end=10,
         convert_to_iter_based=True)
]

default_hooks = dict(
    dict(checkpoint=dict(by_epoch=True, interval=1)),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

default_scope = 'mmdet'  # 默认的注册器域名，默认从此注册器域中寻找模块。请参考 https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/registry.html

env_cfg = dict(
    cudnn_benchmark=False,  # 是否启用 cudnn benchmark
    mp_cfg=dict(  # 多进程设置
        mp_start_method='fork',  # 使用 fork 来启动多进程。'fork' 通常比 'spawn' 更快，但可能存在隐患。请参考 https://github.com/pytorch/pytorch/issues/1355
        opencv_num_threads=0),  # 关闭 opencv 的多线程以避免系统超负荷
    dist_cfg=dict(backend='nccl'),  # 分布式相关设置
)
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        save_dir=
        'temp',
        init_kwargs=dict(
            project='faster_rcnn',
            name='faster_rcnn_01',),
        define_metric_cfg=None,
        commit=True,
        log_code_name=None,
        watch_kwargs=None)
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            type='WandbVisBackend',
            save_dir=
            'temp',
            init_kwargs=dict(
                project='faster_rcnn',
                name='faster_rcnn_01',
                ),
            define_metric_cfg=None,
            commit=True,
            log_code_name=None,
            watch_kwargs=None),
    ],
    name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False
