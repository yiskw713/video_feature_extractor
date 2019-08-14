env CUDA_VISIBLE_DEVICES=1 python extract.py \
/home/cvrg/Documents/dataset/msr-vtt/train_jpg \
/home/cvrg/Documents/dataset/msr-vtt/features/r50_k700_16f \
resnet50 ./weights/resnet50_kinetics700.pth &

env CUDA_VISIBLE_DEVICES=2 python extract.py \
/home/cvrg/Documents/dataset/msr-vtt/test_jpg \
/home/cvrg/Documents/dataset/msr-vtt/features/r50_k700_16f \
resnet50 ./weights/resnet50_kinetics700.pth

env CUDA_VISIBLE_DEVICES=1 python extract_ms.py \
/home/cvrg/Documents/dataset/msr-vtt/train_jpg \
/home/cvrg/Documents/dataset/msr-vtt/features/sfnl152_k700_16f \
slowfast152_nl ./weights/slowfast152_nl_kinetics700.pth &

env CUDA_VISIBLE_DEVICES=2 python extract_ms.py \
/home/cvrg/Documents/dataset/msr-vtt/test_jpg \
/home/cvrg/Documents/dataset/msr-vtt/features/sfnl152_k700_16f \
slowfast152_nl ./weights/slowfast152_nl_kinetics700.pth