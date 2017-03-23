for iterations in 1 2 3 4 5 6 7 8 9 10
do
	for lo in 'surrogate_hinge' 'surrogate_hinge_squares' 'surrogate_squares' 'surrogate_exponential' 'surrogate_sigmoid' 'surrogate_logistic' 'surrogate_double_hinge'
	do
		python3 train_cifar.py -data_dir './data' -save_dir './save' -export_file 'test.csv' -loss $lo -batch_size 128 -max_steps 150000 -num_blocks 3 -learn_rate .01
	done 
done 

