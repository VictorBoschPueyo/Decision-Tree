from node import *

class Tree:
    def __init__(self, data, candidats, test):
        self.data = data
        self.candidats = candidats
        self.test = test
        self.node_arrel = Node("Arrel", None, self.data, self.candidats[:-1])
    
    def calcular_predict(self):
        return self.predict_rec(self.test, self.node_arrel) / self.test.shape[0]

    def predict_rec(self, x_test, node):
        if node.fulla:
            # Calculate right ones
            y = x_test[:,-1]
            x = np.count_nonzero(y == node.decisio)
            return x
        else:
            suma = 0
            atrib = np.where(self.candidats == node.atribut)[0][0]
            for i, fill in enumerate(node.etiquetes_fills):
                nou_data = x_test[x_test[:,atrib] == fill]
                suma += self.predict_rec(nou_data, node.list_fills[i])
            return suma
    
    def expand_tree(self, tipus = 0): 
        # Expand according to the chosen attribute set attribute to choose expansion method
        pila = [self.node_arrel]
    
        while (len(pila) != 0):
            llista_aux = []
            node = pila[0]
            del pila[0]
            
            best_atrib = node.calculate_best_atribute(self.candidats, tipus)
            if best_atrib == -1: # case all rows are equal
                valors_unics, count_valors = np.unique(node.data[:,-1], return_counts=True)
                node.decisio = valors_unics[np.argmax(count_valors)]
                node.fulla = True
                continue
            
            node.atribut = self.candidats[best_atrib][0]
            node.etiquetes_fills = np.unique(node.data[:,best_atrib])
            for fill in node.etiquetes_fills:
                nou_data = node.data[node.data[:,best_atrib] == fill]
                cand = np.delete(node.candidats, np.where(node.candidats == self.candidats[best_atrib])[0])
                nou_node = Node(str(fill), node, nou_data, cand)
                node.list_fills.append(nou_node)
                
                if (nou_node.entropy < 0.3 or nou_node.data.shape[0] < 20 or nou_node.candidats.shape[0] == 0):
                    nou_node.fulla = True
                    valors_unics, count_valors = np.unique(node.data[:,-1], return_counts=True)
                    nou_node.decisio = valors_unics[np.argmax(count_valors)]
                else:
                    llista_aux.append(nou_node)
            pila = llista_aux + pila
