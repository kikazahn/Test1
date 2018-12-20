# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 15:17:41 2018

@author: Kirill
"""

models = (model1, model2, model3)
weights = ('model1.h5','model2.h5','model3.h5')
path = 'D:/MTS_WORK/MNIST/Ensemble/'

for i in range(3):
    model.load_weights(path + weights[i])
    score = model.evaluate(X_test, Y_test, verbose=0)
    
    P3 = model.predict(X_test)
    
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

k = 0
t = np.zeros((10000,10))
tt = np.zeros(10000)
for i in range(10000):
    t1 = P1[i]
    t2 = P2[i]
    t3 = P3[i]
    
    for j in range(10):
        t[i][j] = np.max((t1[j], t2[j], t3[j]))
        
    tt[i] = np.argmax(t[i])
    
    if y_test[i] == tt[i]:
        k += 1
    
print(k/10000)
    
    
    
    
    
    
    
    