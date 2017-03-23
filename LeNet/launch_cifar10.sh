
# cd ~/path

# " Boucle sur le learning rate"
for l_rate in 1e-4 
do
	# "Boucle sur batch_size"
	for b_size in 20
	do
		# "Boucle sur lambda (coefficient de régularisation)"
		for lamb in 0
		do
			# "Boucle sur la perte"
			for lo in 'large_margin_entropy'
			do
			# echo $l_rate $b_size $lamb $lo
			python3 neural_network_v3_with_shuffle.py -data 'cifar10' -loss $lo -batchsize $b_size -nb_iter 80000 -lambda $lamb -learn_rate $l_rate
			done
		done
	done
done




