# Streaming-LDA: A Copula-based Approach to Modeling Topic Dependencies in Document Streams
#(C) Copyright 2016, Hesam Amoualian


#please use the bibtex inside the code to cite this work:

# needs to have toy_dataset.text and vocabulary.py in same path


# You can run the code simply by:
# python Coplda.py [path2train] [Number of topics] [Number of iterations] [Removing stop words] [percent of train documents]
# For instance:
# python Coplda.py toy_dataset.txt 20 50 false 90
# Output:
# 1-cosine(Number of topics).txt   : 1- cosine similarity between topic distributions of two consecutive documents (for example second row number is between doc0 and doc1) (near to zero means more similar)
#lambda(Number of topics).txt   : lambda hyperparameter between topic distributions of two consecutive documents (for example second row number is between doc0 and doc1) (the higher lambda the more similar documents)
#Perplexities(Number of topics).txt  (each row for each iteration)
#topc10words.txt     (top-10 words , each row related to one topic)
