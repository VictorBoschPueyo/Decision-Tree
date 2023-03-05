import pandas as pd
import time

from tree import *

def representacio(node, indent, file, tipus = 0):
    acc = ((indent-1) * "    |")         
    if indent>0:
        acc += "    |"         
    acc += 3 * "-"           
    acc += "Value: " + node.etiqueta + " --> "

    if node.fulla:
        file.write(acc + "Solution:\"" + str(node.decisio) + "\"" + "\n")
    else:
        file.write(acc + "Attribute: " + str(node.atribut) + "\n")
        for fill in node.list_fills:
            representacio(fill, indent+1, file)
            
def load_dataset(path):
    dataset = pd.read_csv(path, header=None, delimiter=',')
    
    return dataset

def tractament_nulls(data):
    for i in range(data.shape[1]):
        col = data[:,i]
        maxim = col[np.argmax(data[:,i])]
        data[:,i] = np.where(col==' ?', maxim, col) 
        
    return data

def kfold(dataset, n_particions, labels):
    """DIVIDE DATASET IN N PARTITIONS"""
    new_dataset = dataset[np.random.permutation(len(dataset))]
    conjunts = []
    for inc,i in enumerate(range(0,new_dataset.shape[0],int(new_dataset.shape[0]/n_particions))):
        if inc < n_particions:
            seguent = (int(new_dataset.shape[0]/n_particions))*(inc+1)
            conjunts.append(new_dataset[i:seguent,:])
    """EXECUTE CROSS-VALIDATION"""
    mitj = 0
    for i, test in enumerate(conjunts):
        conjunts.pop(i)
        train = np.concatenate(conjunts)
        
        arbre = Tree(train, labels, test)
        arbre.expand_tree()
        #########
        pred = arbre.calcular_predict()
        mitj += pred
        print("Predicció per al conjunt", i, ":", pred)
        
        conjunts.insert(i, test)
    
    print("--------------------------")
    print("Resultat mitjà del cross-validation:", mitj / n_particions)
    

def main():
    # Load dataset
    adult_data = load_dataset('/data/adult.data')
    test_data = load_dataset('/data/adult.test')
    adult_names = load_dataset('/data/labels.txt')
    
    # Delete null values
    adult_names = adult_names.dropna()
    
    # Manage null values
    adult_data = pd.DataFrame(tractament_nulls(adult_data.values))
    test_data = pd.DataFrame(tractament_nulls(test_data.values))
    
    # Standardize data
    test_data.iloc[:,-1] = test_data.iloc[:,-1].replace({' <=50K.':' <=50K', ' >50K.':' >50K'}, regex=True)
    
    # Discretize continuous values
    columnes_discr = [0 , 2, 4, 10, 11, 12] # Columns to discretize
    for i in columnes_discr:
        col_train = pd.qcut(adult_data[i], q=4, precision = 0, duplicates="drop")
        col_train = col_train.astype('category')
        col_train = col_train.cat.codes
        
        adult_data[i] = col_train.to_numpy()
        ######################################
        col_test = pd.qcut(test_data[i], q=4, precision = 0, duplicates="drop")
        col_test = col_test.astype('category')
        col_test = col_test.cat.codes
        
        test_data[i] = col_test.to_numpy()
        
    print("==========================")
    print("PREDICCIÓ AMB TRAIN I TEST")
    print("==========================")
    start = time.time()
    arbre = Tree(adult_data.values, adult_names.values, test_data.values)
    
    ## Example using ID3
    arbre.expand_tree() 
        
    end = time.time()
    print("Resultat de la prediccio:", arbre.calcular_predict())
    print("--------------------------")
    print("Temps transcorregut:", end - start)
    with open('/representation/arbre_ID3.txt', 'w') as file:
        representacio(arbre.node_arrel, 0, file)
        
    print("==========================")
    print("==== CROSS-VALIDATION ====")
    print("==========================")
    start = time.time()
    kfold(adult_data.values, 5, adult_names.values)
    end = time.time()
    print("--------------------------")
    print("Temps transcorregut:", end - start)
    print("==========================")
        
    

if __name__ == "__main__":
    main()