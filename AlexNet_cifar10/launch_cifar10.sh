
# cd ~/path

for lo in 'entropy' 'hinge' 'crammer' 'lee' 'surrogate' 'largemargin' 
do
# echo $l_rate $b_size $lamb $lo
python3 cifar10_train.py --loss $lo --max_steps=11
done




