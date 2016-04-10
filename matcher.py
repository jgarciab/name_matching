import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import LSHForest
import pickle
import os

class Matcher():
    def __init__(self,folderSaveData):
        #create directory to save data
        self.folderSaveData = folderSaveData
        if not os.path.exists(folderSaveData):
            os.makedirs(folderSaveData)
            
    
    def tokenize(self,listNames,lenToken=3):
        #string to tokens of size 3
        return [' '.join([name[i:i+lenToken] for i in range(len(name)-2)]) for name in listNames]

    def create_TDIDF(self,trainGrams):
        #Create TDIDF from n-tokens
        TF = TfidfVectorizer()
        tfidfs = TF.fit_transform(trainGrams)
        return TF,tfidfs


    def create_tree(self,listNames,variableName):
        #LSHForest. only once for the main database
        lshf = LSHForest(n_estimators=50,n_candidates=500)
        TF, tfidfs = self.create_TDIDF(self.tokenize(listNames))
        lshf.fit(tfidfs)        
        pickle.dump(lshf,open("{0}/{1}_lshf.dump".format(self.folderSaveData,variableName),"wb+"))
        pickle.dump(listNames,open("{0}/{1}_listNames.dump".format(self.folderSaveData,variableName),"wb+"))
        pickle.dump(TF,open("{0}/{1}_TF.dump".format(self.folderSaveData,variableName),"wb+"))
        

    def match_data_tree(self,variableList,list_names_to_match):
        
        for variableName in variableList:
            with open(self.folderSaveData+variableName+"_matched.csv","w+") as fOUT:
                print("Number of names to match", len(list_names_to_match))

                lshf = pickle.load(open("{0}/{1}_lshf.dump".format(self.folderSaveData,variableName),"rb"))
                TF = pickle.load(open("{0}/{1}_TF.dump".format(self.folderSaveData,variableName),"rb"))
                listNames = pickle.load(open("{0}/{1}_listNames.dump".format(self.folderSaveData,variableName),"rb"))
                tokenMatch = self.tokenize(list_names_to_match)
                tdidf_transformed = TF.transform(tokenMatch)


                print("Finding neighbors")
                distances_, indices_ = lshf.kneighbors(tdidf_transformed,n_neighbors=100)


                for i,name_to_match in enumerate(list_names_to_match):
                    if i%1000 == 0: print("Number matched ", i)
                    distances = distances_[i,:]
                    indices = indices_[i,:]

                    name,distances,indices = self.filter_data_exact(name_to_match,distances,indices)
                    names_matched = [listNames[index] for index in indices]
                    string_to_save = "{0}\t{1}\n".format(name,"\t".join([str(_[0])+"\t"+str(_[1]) for _ in zip(names_matched,distances)]))
                    fOUT.write(string_to_save)
                
    def filter_data_exact(self,name_match,distances,indices,trainingData = None):
        #still to code, for now taking the top 10 matches        
        if trainingData: 
            return name_match, distances[:10],indices[:10] #fancy stuff goes here
        else:
            return name_match, distances[:10],indices[:10]


M = Matcher("./directory_name/")

if 1: M.create_tree(list_names_train,"name_to_save")
M.match_data_tree(["directorsNamesCompaniesNL"],list_names_match)
