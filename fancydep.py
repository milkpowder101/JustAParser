# Use more training examples (check out the 'en.tr' file as opposed to 'en.tr100')
# better features to predict the next parsing action.
# Dynamic oracle
# stacktop  word
# buffertop word
# stacktop  pos
# buffertop pos
# pairword
# pairpos
# distance:      stacktop-buffertop distance
# left-valency:  the number of left  modifiers
# right-valency: the number of right  modifiers
# Unigram
# Third-order features
# label:         Set of dependency labels stacktop-buffertop
# based on graphparser
# http://www.ciml.info/dl/v0_8/ciml-v0_8-ch03.pdf    AVG perceptron
import sys
import networkx as nx

class Weights(dict):
    def __getitem__(self, idx):
        if self.has_key(idx):
            return dict.__getitem__(self, idx)
        else:
            return 0.

    def dotProduct(self, x):
        dot = 0.
        for feat, val in x.iteritems():
            if feat != 'transition':
                dot += val * self[feat]
        return dot

    def update(self, x, y):
        for feat, val in x.iteritems():
            if val != 0. and feat != 'transition':
                self[feat] += y * val

    def minus(self, x, y, c):
        for feat, val in x.iteritems():
            self[feat] = val - (y[feat] / c)

    def sum(self, z):
        # for feat, val in z.iteritems():
        #     self[feat] += z[feat]
        pass

    def avg(self, avgCounter):
        # for feat, val in self.iteritems():
        #     self[feat] = val / avgCounter
        # return self
        pass

def iterCoNLL(filename):

    f = open(filename, 'r')
    G = None
    nn = 0
    for l in f:
        l = l.strip()
        if l == "":
            if G != None:
                yield G
            G = None
        else:
            if G == None:
                nn = nn + 1
                G = nx.Graph()
                G.add_node(0,
                           {'word': '*root*', 'lemma': '*root*', 'cpos': '*root*', 'pos': '*root*', 'feats': '*root*'})
            [id, word, lemma, cpos, pos, feats, head, drel, phead, pdrel] = l.split('\t')
            G.add_node(int(id), {'word': word,
                                 'lemma': lemma,
                                 'cpos': cpos,
                                 'pos': pos,
                                 'feats': feats,
                                 'head': head,
                                 'drel': drel,
                                 'phead': phead,
                                 'pdrel': pdrel})
    if G != None:
        yield G
    f.close()
def iterCoNLLTest(filename):
    f = open(filename, 'r')
    G = None
    nn = 0
    for l in f:
        l = l.strip()
        if l == "":
            if G != None:
                yield G
            G = None
        else:
            if G == None:
                nn = nn + 1
                G = nx.Graph()
                G.add_node(0,
                           {'word': '*root*', 'lemma': '*root*', 'cpos': '*root*', 'pos': '*root*', 'feats': '*root*'})
            [id, word, lemma, cpos, pos, feats, head, drel, phead, pdrel] = l.split('\t')
            G.add_node(int(id), {'word': word,
                                 'lemma': lemma,
                                 'cpos': cpos,
                                 'pos': pos,
                                 'feats': feats,
                                 'head': '_',
                                 'drel': drel,
                                 'phead': phead,
                                 'pdrel': pdrel})
    if G != None:
        yield G
    f.close()

def getFeature(Graph, stack, buffer):
    if(len(stack))==0:
        f1='*root*'
        f3='*root*'
        f19='*root*'
    else:
        f1=Graph.node[stack[-1]]['word']
        f3=Graph.node[stack[-1]]['cpos']
        f19=Graph.node[stack[-1]]['pos']
    if(len(buffer))==0:
        f2='Empty'
        f4='Empty'
        f20='Empty'
    else:
        f2=Graph.node[buffer[0]]['word']
        f4=Graph.node[buffer[0]]['cpos']
        f20=Graph.node[buffer[0]]['pos']
    # f5=[f1,f2]
    # f6=[f3,f4]

    if len(buffer)>1:
        f7=Graph.node[buffer[1]]['word']
        f8=Graph.node[buffer[1]]['cpos']
        f21=Graph.node[buffer[1]]['pos']
    else:
        f7='N_1W'
        f8='N_1P'
        f21='N_1P,pos'

    if len(stack)>1:
        f9=Graph.node[stack[-2]]['word']
        f10=Graph.node[stack[-2]]['cpos']
        f22=Graph.node[stack[-2]]['pos']
    else:
        f9='S_1W'
        f10='S_1P'
        f22='S_1P,pos'

    if len(buffer)>2:
        f15=Graph.node[buffer[2]]['word']
        f16=Graph.node[buffer[2]]['cpos']
        f23=Graph.node[buffer[2]]['pos']
    else:
        f15='N_2W'
        f16='N_2P'
        f23='N_2P,pos'

    if len(stack)>2:
        f17=Graph.node[stack[-3]]['word']
        f18=Graph.node[stack[-3]]['cpos']
        f24=Graph.node[stack[-3]]['pos']
    else:
        f17='S_2W'
        f18='S_2P'
        f24='S_2P,pos'


    # f11:distance - if it is empty???
    f11=str(int(stack[-1])-int(buffer[0]))

    feats = {'sword=' + f1: 1., 
             'bword=' + f2: 1.,
             'scpos=' + f3: 1., 
             'bcpos=' + f4: 1.,
             'w_pair=' + f1 + '_' + f2: 1.,
             'p_pair=' + f3 + '_' + f4: 1.,


             'N_1_W=' + f7: 1.,
             'N_1_P=' + f8: 1.,
             'S_1_W=' + f9: 1.,
             'S_1_P=' + f10: 1.,
             'N_2_W=' + f15: 1.,
             'N_2_P=' + f16: 1.,
             'S_2_W=' + f17: 1.,
             'S_2_P=' + f18: 1.,
             'S1_p,S0_p,N0_p=' + f10 + '_' + f3 + '_' + f4: 1.,
             'S0_p,N0_P,N1_p=' + f3  + '_' + f4 + '_' + f8: 1.,
             'N0_p,N1_p,N2_p=' + f4  + '_' + f8 + '_' + f16:1.,
             'S2_p,S1_p,S0_p=' + f18 + '_' + f10+ '_' + f3: 1.,
             # 'distance=' + f11: 1.
             'S0_w,N0_w,d=' + f1 + '_' + f2 + '_' + f11: 1.,
             'S0_p,N0_p,d=' + f3 + '_' + f4 + '_' + f11: 1.,

             'spos=' + f19: 1.,
             'bpos=' + f20: 1.,
             'p_pair=' + f19 + '_' + f20: 1.,
             'p_pair=' + f3 + '_' + f20: 1.,
             'p_pair=' + f19 + '_' + f4: 1.,


             'N1_w,S1_w='+ f7 + '_' + f9: 1.,
             'N1_p,S1_p='+ f8 + '_' + f10: 1.,
             'N1_w,N1_p='+ f7 + '_' + f8: 1.,
             'S1_w,S1_p='+ f9 + '_' + f10: 1.,

             'N2_w,S2_w='+ f15 + '_' + f17: 1.,
             'N2_p,S2_p='+ f16 + '_' + f18: 1.,
             'N2_w,N2_p='+ f15 + '_' + f16: 1.,
             'S2_w,S2_p='+ f17 + '_' + f18: 1.,

             'N_1_P,pos=' + f21: 1.,
             'S_1_P,pos=' + f22: 1.,
             'N_2_P,pos=' + f23: 1.,
             'S_2_P,pos=' + f24: 1.,

             'S1_p,S0_p,N0_p,pos=' + f22 + '_' + f19 + '_' + f20: 1.,
             'S0_p,N0_P,N1_p,pos=' + f19  + '_' + f20 + '_' + f21: 1.,
             'N0_p,N1_p,N2_p,pos=' + f20  + '_' + f21 + '_' + f23:1.,
             'S2_p,S1_p,S0_p,pos=' + f24 + '_' + f22 + '_' + f19: 1.,

             'N0_w,N1_w='+ f2 + '_' + f7: 1.,
             'N0_p,N1_p='+ f4 + '_' + f8: 1.,
             'S0_w,S1_w='+ f1 + '_' + f9: 1.,
             'S0_p,S1_p='+ f3 + '_'+ f10: 1.,

             'N1_w,N2_w='+ f7 + '_' + f15: 1.,
             'N1_p,N2_p='+ f8 + '_' + f16: 1.,
             'S1_w,S2_w='+ f9 + '_' + f17: 1.,
             'S1_p,S2_p='+ f10 + '_' +f18: 1.,

             'N0_p,N1_p,pos='+ f20 + '_' + f21: 1.,
             'S0_p,S1_p,pos='+ f19 + '_' + f22: 1.,

             'N1_p,N2_p,pos='+ f21 + '_' + f23: 1.,
             'S1_p,S2_p,pos='+ f22 + '_' + f24: 1.,

             'N0_w,N1_w,N2_w='+f2 + '_' + f7 + '_' + f15: 1.,
             'S0_w,S1_w,S2_w='+f1 + '_' + f9 + '_' + f17: 1.,

             'S0_w,d='+f1 + '_' + f11: 1.,
             'S0_p,d='+f3 + '_' + f11: 1.,
             'N0_w,d='+f2 + '_' + f11: 1.,
             'N0_p,d='+f4 + '_' + f11: 1.


            }




    return feats


def update(weights, weightSum, trueGraph, predictGraph, error):
    global counter
    for i, j in trueGraph.edges_iter():
        if trueGraph[i][j]['transition'] != predictGraph[i][j]['transition']:
            feats = trueGraph.get_edge_data(i,j)['Feature']
            truth = trueGraph[i][j]['transition']
            weightSum[truth].update(feats, counter)
            weights[truth].update(feats, 1)
            wrong = predictGraph[i][j]['transition']
            weights[wrong].update(feats, -1)
            weightSum[wrong].update(feats, -counter)
            error += 1
        else:
            pass
        counter=counter+1
    return error


def output(predGraph, f):
    d = {}
    for (i, j) in predGraph.edges_iter():
        if predGraph[i][j]['transition'] == 'left':
            d[i] = j
        elif predGraph[i][j]['transition'] == 'right':
            d[j] = i

    for i in predGraph.nodes():
        if i == 0:
            continue
        if i not in d:
            d[i] = 0
        g = predGraph.node[i]
        out = [str(i), g['word'], g['lemma'], g['cpos'], g['pos'],g['feats'], str(d[i]), g['drel'], g['phead'], g['pdrel']]
        f.write('\t'.join(out) + '\n')
    f.write('\n')


def runOneExample(weights, weightSum, trueGraph, error):
    predictGraph = nx.Graph()
    Stack = [0]
    Buffer = list(range(1, len(trueGraph.node)))
    d = {}
    for i in trueGraph.nodes():
        if i != 0:
            g = trueGraph.node[i]
            d[i] = int(g['head'])
    while len(Buffer) != 0:
        # prevent stack is empty
        if (len(Stack) == 0):
            Stack.append(Buffer.pop(0))
            if len(Buffer) == 0:
                break
        bufferTop = Buffer[0]
        stackTop = Stack[-1]
        # find if there any further use of right, prevent it if need.
        flag = True
        for i in range(1, len(Buffer)):
            if d[Buffer[i]] == bufferTop:
                flag = False
                break
        feats = getFeature(trueGraph,Stack, Buffer)
        if d[bufferTop] == stackTop and flag:
            trueGraph.add_edge(stackTop, bufferTop, {'transition': 'right'})
            trueGraph.add_edge(stackTop, bufferTop, {'Feature': feats})

            Buffer.pop(0)
            Buffer.insert(0, Stack.pop(-1))
        elif stackTop != 0 and d[stackTop] == bufferTop:
            trueGraph.add_edge(stackTop, bufferTop, {'transition': 'left'})
            trueGraph.add_edge(stackTop, bufferTop, {'Feature': feats})

            Stack.pop(-1)
        else:
            trueGraph.add_edge(stackTop, bufferTop, {'transition': 'shift'})
            trueGraph.add_edge(stackTop, bufferTop, {'Feature': feats})

            Stack.append(Buffer.pop(0))

    #true graph with edge


        leftWeight = weights['left'].dotProduct(feats)
        rightWeight = weights['right'].dotProduct(feats)
        shiftWeight = weights['shift'].dotProduct(feats)
        prediction = max(leftWeight, rightWeight, shiftWeight)
        if prediction==shiftWeight:
            predictGraph.add_edge(stackTop, bufferTop, {'transition': 'shift'})
        elif prediction==leftWeight and stackTop != 0:
            predictGraph.add_edge(stackTop, bufferTop, {'transition': 'left'})
        else:
            predictGraph.add_edge(stackTop, bufferTop, {'transition': 'right'})


    error= update(weights, weightSum,  trueGraph, predictGraph, error)
    return  error


def predict(weights, G, f):
    Stack = [0]
    Buffer = list(range(1, len(G.nodes())))

    while len(Buffer) != 0:
        if len(Stack) == 0:
            Stack.append(Buffer.pop(0))
            if len(Buffer) == 0:
                break
        bufferTop = Buffer[0]
        stackTop = Stack[-1]
        feats = getFeature(G, Stack, Buffer)
        leftWeight = weights['left'].dotProduct(feats)
        rightWeight = weights['right'].dotProduct(feats)
        shiftWeight = weights['shift'].dotProduct(feats)

        prediction=max(leftWeight,rightWeight,shiftWeight)
        if shiftWeight==prediction and len(Buffer) > 1:
            G.add_edge(stackTop, bufferTop, {'transition': 'shift'})
            Stack.append(Buffer.pop(0))

        elif leftWeight==prediction and stackTop != 0:
            G.add_edge(stackTop, bufferTop, {'transition': 'left'})
            Stack.pop()
        else:
            G.add_edge(stackTop, bufferTop, {'transition': 'right'})
            Buffer.pop(0)
            Buffer.insert(0, Stack.pop())

    output(predGraph, f)

if __name__ == '__main__':
    train = sys.argv[1]
    test = sys.argv[2]
    test_output = sys.argv[3]


    weights = {'left': Weights(), 'right': Weights(), 'shift': Weights()}
    weightSum = {'left': Weights(), 'right': Weights(), 'shift': Weights()}

    averageWeights = {'left': Weights(), 'right': Weights(), 'shift': Weights()}
    counter = 0
    for iteration in range(10):
        error = 0

        for trueGraph in iterCoNLL(train):
             error = runOneExample(weights, weightSum, trueGraph, error)

        print error

        f = open(test_output, 'w')

        averageWeights['left'].minus(weights['left'], weightSum['left'],counter)
        averageWeights['right'].minus(weights['right'], weightSum['right'],counter)
        averageWeights['shift'].minus(weights['shift'], weightSum['shift'],counter)


        for predGraph in iterCoNLL(test):
            predict(averageWeights, predGraph, f)
        f.close()