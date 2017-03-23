for iterations in 1 2 3 4 5 6 7 8 9 10
do
	for lo in 'cross_entropy' 'hinge' 'crammer' 'lee' 'GEL' 'GLL'
	do
		python3 train_cifar.py -data_dir './data' -save_dir './save' -export_file 'test.csv' -loss $lo -batch_size 128 -max_steps 150000 -num_blocks 3 -learn_rate .01
	done 
done 
