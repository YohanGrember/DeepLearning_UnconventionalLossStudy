The python file you can use to train ResNet on CIFAR-10 is train_cifar.py
It was built to work on python3.
Last edited: 23/03/2017

optional arguments:

  -h, --help            show this help message and exit
  
  -data_dir DATA_DIR, --data_dir DATA_DIR
                        Directory for storing the datasets
                        
  -save_dir SAVE_DIR, --save_dir SAVE_DIR
                        Directory for writing event logs and checkpoints
                        
  -export_file EXPORT_FILE, --export_file EXPORT_FILE
                        name of the csv export file
                        
  -loss LOSS, --loss LOSS
                        loss function : cross_entropy / weston / crammer / lee /
                        surrogate_hinge / surrogate_hinge_squares / surrogate_squares /
                        surrogate_exponential / surrogate_sigmoid / surrogate_logistic /
                        surrogate_saturated_hinge / GEL / GLL / large_margin 
                        
  -batch_size BATCH_SIZE, --batch_size BATCH_SIZE
                        batch size
                        
  -max_steps MAX_STEPS, --max_steps MAX_STEPS
                        nombre diterations
                        
  -num_blocks NUM_BLOCKS, --num_blocks NUM_BLOCKS
                        6n+2 total weight layers will be used. num_blocks = 3
                        : ResNet-20. num_blocks = 5 : ResNet-32. num_blocks =
                        8 : ResNet-50. num_blocks = 18 : ResNet-110
                        
  -learn_rate LEARNING_RATE, --learning_rate LEARNING_RATE
                        learning rate for gradient descent
                        
  -m M, --m M           margin coefficient for Large Margin Softmax Loss
  
  -load LOAD, --load LOAD
                        Initialize the network from a checkpoint ?
                        
  -load_dir LOAD_DIR, --load_dir LOAD_DIR
                        Directory from which to load the network
