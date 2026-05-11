"""
主入口: 一键训练+推理
智海算法调优 - 天气四分类任务

使用方法:
    # 训练
    python main.py --mode train

    # 推理
    python main.py --mode inference --ckpt output/checkpoints/best_model.pth

    # 训练+推理
    python main.py --mode all
"""

import os
import argparse
import sys

# 确保项目根目录在path中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="天气分类 - 智海算法调优")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["train", "inference", "all"],
                        help="运行模式")
    parser.add_argument("--model", type=str, default="efficientnet_b0",
                        help="模型选择")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="推理用权重路径")
    parser.add_argument("--test-dir", type=str, default=None,
                        help="测试集目录")
    parser.add_argument("--tta", action="store_true", default=True,
                        help="测试时增强")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="覆盖默认batch size")
    parser.add_argument("--epochs", type=int, default=None,
                        help="覆盖默认训练轮数")
    parser.add_argument("--lr", type=float, default=None,
                        help="覆盖默认学习率")

    args = parser.parse_args()

    # 覆盖配置
    if args.model:
        import config
        config.MODEL_NAME = args.model
    if args.batch_size:
        import config
        config.BATCH_SIZE = args.batch_size
    if args.epochs:
        import config
        config.NUM_EPOCHS = args.epochs
    if args.lr:
        import config
        config.LEARNING_RATE = args.lr

    # ---- 数据目录检查 ----
    import config
    print("="*60)
    print("天气分类 - 智海算法调优")
    print(f"模型: {config.MODEL_NAME}")
    print(f"设备: {config.DEVICE}")
    print(f"数据目录: {config.DATA_DIR}")
    print("="*60)

    if not os.path.exists(config.DATA_DIR):
        print("\n[提示] 数据目录不存在，请创建以下结构:")
        print(f"  {config.DATA_DIR}/")
        print(f"    train/")
        print(f"      sunny/    <- 晴天图片")
        print(f"      rainy/    <- 雨天图片")
        print(f"      cloudy/   <- 阴天图片")
        print(f"      snowy/    <- 雪天图片")
        print(f"    val/        <- (可选)验证集，同train结构")
        print(f"    test/       <- (可选)测试集")
        if args.mode != "inference":
            print("\n请准备好数据后再运行训练!")
            return

    # ---- 执行 ----
    if args.mode in ("train", "all"):
        print("\n>>> 开始训练...")
        from train import train
        model, ckpt_path = train()
        print(f">>> 训练完成! 最佳模型: {ckpt_path}")

    if args.mode in ("inference", "all"):
        print("\n>>> 开始推理...")
        from inference import inference_and_submit
        inference_and_submit(
            ckpt_path=args.ckpt,
            test_dir=args.test_dir,
            use_tta=args.tta,
        )

    print("\n>>> 全部完成!")


if __name__ == "__main__":
    main()
