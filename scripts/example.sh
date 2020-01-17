python utils/build_dataset.py msr-vtt /media/cvrg/ssd2t2/msr-vtt/hdf5
python extract.py /media/cvrg/ssd2t2/msr-vtt/hdf5 /media/cvrg/ssd2t2/msr-vtt/features \
./csv/msr-vtt.csv slowfast_nl ./weights/slowfast152_nl_kinetics700.pth