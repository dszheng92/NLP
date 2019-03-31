The main script is the word2vec.py file and it is python2 script. 
After setting the parameter insides the script, by 'python word2vec.py', you will have the performance on dev data and a simiarity file of test data.
The explannation of parameter is in the file and also attached here.


Parameters in the script:
	## Generate training data, you need to give:
		# Data path, 
		# Filter_number to get rid of words of low frequence
		# Context mode, dep for dependency, bow for linear bag of words
		# Window size for bag of words context model
		# CONLL data path for dependency context mode
		# subsample threshold size that reduces the high frequency words number to threshold
	train_int,vocab,vocab_count,context_fre,word_size,context_size,window,subsample_n = load_train_data(data_path='./data/training/training-data.1m',filter_number=10,mode='bow',window=2,Conll='./data/training/training-data.1m.conll',subsample_n=2000)
	## Training model and output reuslts
		# Embed_size, embedding dimension size, default is 300
		# Batch_size, default is 300
		# num_sampled for negative sampling, default is 5
		# Epoch of training, default is 6
	model = Word2VecModel(word_size,context_fre, context_size,vocab,window,subsample_n=2000,mode='bow',embed_size=300, batch_size=300,num_sampled=5, epoch=6)
	## If you want to continue precious model, input here
	#mode.pc.populate('xxx.model')
	mode.train(train_int)
	## Provide traget file with word1, word2 and this function calculates similarity
	## and output file given the filename. If label is given, accuracy will be calculated.
	## if label is None, it only generates similarity file (for test).
	mode.evaluate(vocab_count,target='./data/similarity/dev_x.csv',label= './data/similarity/dev_y.csv',filename = 'prediction_dev.csv')
	mode.evaluate(vocab_count,target='./data/similarity/test_x.csv',label= None,filename = 'test_y.csv')
