"(C) Copyright 2016, Hesam Amoualian"
"hesam.amoualian@imag.fr"
# References paper: Streaming-LDA: A Copula-based Approach to Modeling Topic Dependencies in Document Streams
""""@inproceedings{amoualian2016kdd,
 author = {Amoualian, Hesam and Clausel, Marianne and Gaussier, Eric and Amini, Massih-Reza},
 title = {Streaming-LDA: A Copula-based Approach to Modeling Topic Dependencies in Document Streams},
 booktitle = {Proceedings of the 22nd International Conference on Knowledge Discovery and Data Mining},
 series = {SIGKDD},
 year = {2016},
 isbn = {978-1-4503-4232-2},
 location = {San Francisco, California, USA},
 pages = {695--704},
 numpages = {10},
 url = {http://doi.acm.org/10.1145/2939672.2939781},
 doi = {10.1145/2939672.2939781},
 acmid = {2939781},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {copulas, document streams, latent dirichlet allocation, topic dependencies},
}"""






from pylab import *
import numpy, codecs
from scipy.special import gamma, gammaln
from datetime import datetime
import vocabulary
import time
import math
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from itertools import groupby
from scipy.stats import gamma as gamma1
import json
import sys


def copula(x,y,lambdal): # Frank Copula Density
    num=numpy.log(lambdal*numpy.float64(numpy.exp(-lambdal*(x+y)))-lambdal*numpy.float64(numpy.exp(-lambdal*(x+y+1))))
    denum=numpy.log((-numpy.float64(numpy.exp(-lambdal*(x+y)))-numpy.float64(numpy.exp(-lambdal))+numpy.float64(numpy.exp(-lambdal*x))+numpy.float64(numpy.exp(-lambdal*y)))**2)

        
    return numpy.exp(num-denum)




class lda_gibbs_sampling:
    def __init__(self, K=None, alpha=None, beta=None, docs= None, V= None ,percent=None,lastit=None):
        self.K = K
        self.alpha = numpy.ones(K)*alpha # parameter of topics prior
        self.beta = numpy.ones(V)*beta   # parameter of words prior
        self.docs = docs # a list of lists, each inner list contains the indexes of the words in a doc, e.g.: [[1,2,3],[2,3,5,8,7],[1, 5, 9, 10 ,2, 5]]
        self.V = V # how many different words in the vocabulary 
        self.z_m_n = {} # topic assignements for each of the N words in the corpus. N: total number of words in the corpus (not the vocabulary size).
        self.n_m_z = numpy.zeros((len(self.docs), K))     # M documents and Ktopics: number of words assigned to topic z in document m
        self.n_z_t = numpy.zeros((K, V))+beta  # (K topics) x |V| : number of times a word v is assigned to a topic z
        self.theta = numpy.zeros((len(self.docs), K)) # Topic Distribution for whole of Documents
        self.phi = numpy.zeros((K, V)) # topic-word distribution, \phi K topics V (vocab size)
        self.n_z = numpy.zeros(K) + V * beta # number of times words are assigned to a topic z
        self.Z=numpy.zeros((len(self.docs), K)) # Gamma distribution for each Documents with length of K topics
        # self.R = numpy.zeros((K, V)) #Gamma distribution for every topic and words
        # self.LF=numpy.zeros(K) #Mu for dependency between phi
        # self.newphi = numpy.zeros((K, V)) #Temporary for initializing phi for metro
        # self.newR = numpy.zeros((K, V)) # Temporary for initializing R for metro
        self.trainpercent=percent/100
        self.lastit=lastit
        self.inloop=2 #greater than zero
        self.pers=[]  #collecting perplex
        self.conv=False
        
        N=0
        for m, doc in enumerate(docs):         # Initialization of the data structures
            for n,w in enumerate(doc):
                z = numpy.random.randint(0, K) # Randomly assign a topic to a word.                  
                self.n_m_z[m, z] += 1          # increase the number of words assigned to topic z in the m doc.
                self.n_z_t[z,w] += 1          #   number of times a word is assigned to this particular topic
                self.z_m_n[(m,n)]=z           # update the array that keeps track of the topic assignements in the words of the corpus.
                self.n_z[z] += 1
                N+=1
        print 'Total number of words:',N


    def inference(self,iteration):
        lamb=[]
        rlam=[]
        for m, doc in enumerate(self.docs): # For loop for documents
            for Noid in range(0,self.inloop): # iterate over each documents
                olam=0
                al=1
                # self.LF=numpy.zeros(self.K)
                self.Z[m]=gamma1.rvs(self.n_m_z[m]+self.alpha)
                if m<=int(len(self.docs)*self.trainpercent) : # Train documents
                        
                        if m>0 and (Noid>0 or iteration>0): # For other Documents
                            a2=gamma1.cdf(self.Z[m-1],self.n_m_z[m-1]+self.alpha)
                            # olam=numpy.random.uniform(0,1) #Initialize First Lambda for copula  metropolice
                            al=1-cosine_similarity(self.Z[m], self.Z[m-1]) # finding similarity for runing metropolice hasting for Lambda
                            olam=sum(numpy.random.exponential(1/al,100))/100
                            self.Z[m]=gamma1.rvs(self.n_m_z[m]+self.alpha) #Initialize First Gamma distribution for copula  metropolice
                            Nc=0 # Counter for Metropolice hasting for Copula
                            NZ=[] # Temporal Place for Gamma distribution for copula
                            while Nc<5: # metropolice hasting for copula
                                u1 = numpy.random.uniform(0,1) # randomly find initial state for acception rejection
                                NZ=gamma1.rvs(self.n_m_z[m]+self.alpha) # new Gamma distribution
                                denum=copula(gamma1.cdf(self.Z[m-1],self.alpha),gamma1.cdf(self.Z[m],self.alpha),olam)
                                num=copula(gamma1.cdf(self.Z[m-1],self.alpha),gamma1.cdf(NZ,self.alpha),olam)

                            
                                for i in range (self.K): # check for acceptance or rejection and update whole of vector of Z with old Lambda
                                    if denum[i]==0:
                                        a1=1
                                    else:
                                        a1=numpy.minimum(num[i]/denum[i],1)
                                    if u1<a1:
                                        self.Z[m][i]=NZ[i]
                                Nc+=1
                        
                        
                            
                            # al=1-cosine_similarity(self.Z[m], self.Z[m-1]) # finding similarity for runing metropolice hasting for Lambda
                            # Nla=0
                            # sumv=[]
                            # newv=0
                         
                            # while Nla<10: # Loop for Metro police hasting for Lambda
#
#                                 nlam = numpy.random.exponential(1/al, 1) # New lambda for metro algorithm(use one Lambda for all of cordinates of one document)
#                                 if nlam<200 : # this is for makeing limitaion to not have overflow
#                                     u = numpy.random.uniform(0,1) # acceptance rejection parameter
#                                     num=al*numpy.exp(-olam*al)*copula(gamma1.cdf(self.Z[m-1,0],self.alpha[0]),gamma1.cdf(self.Z[m,0],self.alpha[0]),nlam)
#                                     denum=al*numpy.exp(-nlam*al)*copula(gamma1.cdf(self.Z[m-1,0],self.alpha[0]),gamma1.cdf(self.Z[m,0],self.alpha[0]),olam)
#                                     a=numpy.minimum(num/denum,1)
#
#                                     newv=olam
#                                     if u<a : # accapet or reject condition
#                                             olam=nlam
#                                     sumv.append(int((olam-newv)==0))
#                                     Nla +=1
#                                     if Nla>5 : # for making algorithm faster and if we have convergence
#                                         grouped_sumv = [(k, sum(1 for i in g)) for k,g in groupby(sumv)]
#                                         for i in range(0,len(grouped_sumv)):
#                                             if grouped_sumv[i][0]==1:
#                                                 if grouped_sumv[i][1]>4:
#                                                     Nla=10
#                                                     break

                            # for i in range(self.K): #metropolice hasting for finding miu dependency between two sequnetial phi in time for all the topics
#                                 self.newR[i]=gamma1.rvs(self.n_z_t[i]+0.5)
#                                 self.newphi[i]=self.newR[i]/self.newR[i,:].sum() # phi which comes from initializing of R gamma funcation
#                                 cs=cosine_similarity(self.phi[i], self.newphi[i])
#                                 nl=1-cs
#                                 Nla=0
#                                 sumv=[]
#                                 oLF=numpy.random.uniform(0,1)
#                                 while Nla<10: # metropolice hasting algorithm for finding miu(between phi topic i at time t and phi topic i at which is from initializing gamma funcation)
#
#                                     nLF = sum(numpy.random.exponential(1/nl, 100))/100 # find new miu
#                                     u = numpy.random.uniform(0,1) # acceptance rejection parameter
#                                     num=nl*numpy.exp(-oLF*nl)*gamma1.pdf(self.newR[i,0],0.5+nLF*self.phi[i,0])
#                                     denum=nl*numpy.exp(-nLF*nl)*gamma1.pdf(self.newR[i,0],0.5+oLF*self.phi[i,0])
#                                     a=numpy.minimum(num/denum,1)
#                                     if u<a : # acceptance rejecetion condition
#                                         oLF=nLF
#                                     sumv.append(int((oLF-nLF)==0))
#
#                                     Nla +=1
#                                     if Nla>5 : # for checking if Metropolice hasting algorithm converged for miu
#                                         grouped_sumv = [(k, sum(1 for i in g)) for k,g in groupby(sumv)]
#                                         for i in range(0,len(grouped_sumv)):
#                                             if grouped_sumv[i][0]==1:
#                                                 if grouped_sumv[i][1]>4: # check if Metropolice hasting algorithm converged for miu
#                                                     Nla=10
#                                                     break
#                                 self.LF[i]=oLF

                        self.theta[m]=self.Z[m]/sum(self.Z[m]) # finding  Theta according to last updated Z
                        
                        for n,w in enumerate(doc): # run Gibbs sampling and updating Phi and finding New Z from Multinomial distirubution
                           
                           z=self.z_m_n[(m,n)]
                           self.n_m_z[m,z] -=1
                           self.n_z_t[z,w] -=1
                           self.n_z[z] -= 1
                           # self.R[z,w]=gamma1.rvs(self.n_z_t[z,w]+self.LF[z]*self.phi[z,w])
                                
                           # self.phi[z,w]=self.R[z,w]/self.R[z,:].sum()
                           p_z = self.theta[m] * self.n_z_t[:,w]/self.n_z # this is multipling of Theta and Phi function
                           new_z = numpy.random.multinomial(1, p_z/p_z.sum()).argmax()   # One multinomial draw
                           self.n_m_z[m,new_z] +=1
                           self.n_z_t[new_z,w] +=1
                           self.n_z[new_z] += 1
                           self.z_m_n[(m,n)]=new_z
            


                else: # Test data set same as train data set( we just use previous Phi for updating Z and then Update Phi again)
                    if Noid>0 or iteration>0 : # doing metropolice hasting same as before for finding Lambda after first iteration to have initalinzing for time =0
                        a2=gamma1.cdf(self.Z[m-1],self.n_m_z[m-1]+self.alpha)
                        al=1-cosine_similarity(self.Z[m], self.Z[m-1])
                        olam=sum(numpy.random.exponential(1/al,100))/100
                        # olam=numpy.random.uniform(0,1)
                        Nc=0
                        NZ=[]
                        while Nc<5:
                            u1 = numpy.random.uniform(0,1)
                            NZ=gamma1.rvs(self.n_m_z[m]+self.alpha)
                            denum=copula(gamma1.cdf(self.Z[m-1],self.alpha),gamma1.cdf(self.Z[m],self.alpha),olam)
                            num=copula(gamma1.cdf(self.Z[m-1],self.alpha),gamma1.cdf(NZ,self.alpha),olam)
                            if denum[i]==0:
                                a1=1
                            else:
                                a1=numpy.minimum(num[i]/denum[i],1)
                            if u1<a1:
                                self.Z[m][i]=NZ[i]
                            Nc+=1
                    
                    
                    
                        # al=1-cosine_similarity(self.Z[m], self.Z[m-1])
                        # Nla=0
#                         sumv=[]
#                         newv=0
#                         while Nla<10:
#
#                             nlam = numpy.random.exponential(1/al, 1)
#                             if nlam<200 :
#                                 u = numpy.random.uniform(0,1)
#                                 num=al*numpy.exp(-olam*al)*copula(gamma1.cdf(self.Z[m-1,0],self.alpha[0]),gamma1.cdf(self.Z[m,0],self.alpha[0]),nlam)
#                                 denum=al*numpy.exp(-nlam*al)*copula(gamma1.cdf(self.Z[m-1,0],self.alpha[0]),gamma1.cdf(self.Z[m,0],self.alpha[0]),olam)
#                                 a=numpy.minimum(num/denum,1)
#
#                                 newv=olam
#                                 if u<a :
#                                     olam=nlam
#                                 sumv.append(int((olam-newv)==0))
#                                 Nla +=1
#                                 if Nla>5 :
#                                     grouped_sumv = [(k, sum(1 for i in g)) for k,g in groupby(sumv)]
#                                     for i in range(0,len(grouped_sumv)):
#                                         if grouped_sumv[i][0]==1:
#                                             if grouped_sumv[i][1]>4:
#                                                 Nla=10
#                                                 break
                
                        # for i in range(self.K): #metropolice hasting to find miu for all of the topics
#                             self.newR[i]=gamma1.rvs(self.n_z_t[i]+0.5)
#                             self.newphi[i]=self.newR[i]/self.newR[i,:].sum() # phi which comes from initializing of R gamma funcation
#                             cs=cosine_similarity(self.phi[i], self.newphi[i])
#                             nl=1-cs
#                             Nla=0
#                             sumv=[]
#                             oLF=numpy.random.uniform(0,1)
#                             while Nla<10: # metropolice hasting algorithm for finding miu(between phi topic i at time t and phi topic i at which is from initializing gamma funcation)
#
#                                 nLF = sum(numpy.random.exponential(1/nl, 100))/100 # find new miu
#                                 u = numpy.random.uniform(0,1) # acceptance rejection parameter
#                                 num=nl*numpy.exp(-oLF*nl)*gamma1.pdf(self.newR[i,0],0.5+nLF*self.phi[i,0])
#                                 denum=nl*numpy.exp(-nLF*nl)*gamma1.pdf(self.newR[i,0],0.5+oLF*self.phi[i,0])
#                                 a=numpy.minimum(num/denum,1)
#                                 if u<a : # acceptance rejecetion condition
#                                     oLF=nLF
#                                 sumv.append(int((oLF-nLF)==0))
#
#                                 Nla +=1
#                                 if Nla>5 : # for checking if Metropolice hasting algorithm converged for miu
#                                     grouped_sumv = [(k, sum(1 for i in g)) for k,g in groupby(sumv)]
#                                     for i in range(0,len(grouped_sumv)):
#                                         if grouped_sumv[i][0]==1:
#                                             if grouped_sumv[i][1]>4: # check if Metropolice hasting algorithm converged for miu
#                                                 Nla=10
#                                                 break
#                             self.LF[i]=oLF


                    self.theta[m]=self.Z[m]/sum(self.Z[m])
                    for n,w in enumerate(doc):
                        
                        z=self.z_m_n[(m,n)]
                        self.n_m_z[m,z] -=1
                        O=self.n_z_t[:,w]/self.n_z
                        self.n_z_t[z,w] -=1
                        self.n_z[z] -= 1
                        # self.R[z,w]=gamma1.rvs(O+self.LF[z]*self.phi[z,w])
                        
                        # self.phi[z,w]=self.R[z,w]/self.R[z,:].sum()
                        
                        p_z = self.theta[m] * O
                        new_z = numpy.random.multinomial(1, p_z/p_z.sum()).argmax()   # One multinomial draw, for a distribution over topics, with probabilities summing to 1
                        self.n_m_z[m,new_z] +=1
                        self.z_m_n[(m,n)]=new_z
                        self.n_z_t[new_z,w] +=1
                        self.n_z[new_z] += 1
                        O=[]
                
                if Noid>0 and iteration>1: #check the convergence of theta for iteratinf over one document , if got conveged we stop the loop of one document and go to the next
                    dst = distance.euclidean(self.theta[m],Ttheta)
                    if dst<0.15:
                        break
                Ttheta=numpy.copy(self.theta[m]) # for saving last theta dist to comapre with next sample
                
                
            rlam.append(al)
            lamb.append(olam)

# This part is for finding Perpelixty Over Test dataset
        per=0
        b=0
        c=0
        self.phi=self.n_z_t/ self.n_z[:, numpy.newaxis]
        for m, doc in enumerate(self.docs):
            if m>self.trainpercent:
                b+=len(doc)
                
                for n, w in enumerate(doc):
                    l=0
                    for i in range(self.K):
                        l+=self.theta[m,i]*self.phi[i,w]
                    c+=numpy.log(l)
 
        per=numpy.exp(-c/b)
        if iteration>0 and per>self.pers[-1]:
            print "converged"
            self.conv=True
        self.pers.append(per)
        print "perpelixity", per
        if len(self.pers)==self.lastit+1 or self.conv==True :
            numpy.savetxt('Perplexities'+str(self.K)+'.txt',self.pers,fmt='%.2f')
        
        if len(self.pers)==self.lastit+1 or self.conv==True:
            numpy.savetxt('1-cosine'+str(self.K)+'.txt',rlam,fmt='%.2f')
            numpy.savetxt('lambda'+str(self.K)+'.txt',lamb,fmt='%.2f')
        
        return self.conv
            
    def worddist(self):
       
        """topic-word distribution, \phi  (Z topics) x (V words)  """
        return self.phi


if __name__ == "__main__":
    path2input = sys.argv[1]
    Numberoftopic = sys.argv[2]
    Numberofiteration = sys.argv[3]
    having_stopword = json.loads(sys.argv[4])
    percent = json.loads(sys.argv[5])
    corpus = codecs.open(path2input, 'r', encoding='utf8').read().splitlines()
    iterations = int(Numberofiteration)
    voca = vocabulary.Vocabulary(excluds_stopwords=having_stopword)
    
    docs = [voca.doc_to_ids(doc) for doc in corpus]
    print 'whole number of documents:',len(docs)
    print 'Number of train documents:',int(len(docs)*float(percent)/100)
    print 'Number of test documents:',len(docs)-int(len(docs)*float(percent)/100)
    print 'whole number of unique words:',voca.size()
    Number=[]
    for line in docs:
        Number.append(len(line))
    print 'Average number of words per each doc:', sum(Number)/len(Number)
    lda = lda_gibbs_sampling(K=int(Numberoftopic), alpha=0.5, beta=0.5, docs=docs, V=voca.size(),percent=float(percent),lastit=iterations-1)
    
    for i in range(iterations):
        starting = datetime.now()
        print "iteration:", i 
        conv=lda.inference(i)
        print "Took:", datetime.now() - starting
        if conv==True:
            break
	
    d = lda.worddist()
    f=open('topc10words.txt','w')
    for i in range(int(Numberoftopic)):
        ind = numpy.argpartition(d[i], -10)[-10:] # an array with the indexes of the 10 words with the highest probabilitity in the topic
        f.write('topic'+str(i)+': ')
        print 'topic'+str(i)+': ',
        for j in ind:
            print voca[j],
            f.write(voca[j]+' ')
        print 
        f.write('\n')
    f.close()
        
        
        
