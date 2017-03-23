for iterations in 1 2 3 4 5 6 7 8 9 10
do
        for lo in 'large_margin'
        do
                for m in 4 5 6 
                do
                        python3 train_cifar.py -data_dir './data' -save_dir './save' -export_file 'test.csv' -loss $lo -batch_size 128 -max_steps 100000 -num_blocks 3 -learn_rate .01 -m $m 
                done
        done
done
~          
