import pandas as pd
import numpy as np
import math

def add_features():
    df = pd.read_csv('/home/rahul/Desktop/train_final')
    #df = df.sample(frac=1)
    A = df.as_matrix()

    ds = []
    labels = []

    for i in range(A.shape[0]):
        li = []
        if not(math.isnan(A[i,1])) and not(math.isnan(A[i,2])) and not(str(A[i,5])=='nan'):
            li.append(math.floor(A[i,1]))
            li.append(math.floor(A[i,2]))
            logs = A[i,5].split("#")
            click_ct=0
            cart_ct=0
            purchase_ct=0
            fav_ct=0
            days=[]
            purchasedays=[]
            items=[]
            brands=[]
            categories=[]
            for log in logs:
                action = log.split(":")
                if action[4]=='0':
                    click_ct += 1
                elif action[4]=='1':
                    cart_ct += 1
                elif action[4]=='2':
                    purchase_ct += 1
                    purchasedays.append(action[3])
                else:
                    fav_ct +=1
                days.append(action[3])
                items.append(action[0])
                categories.append(action[1])
                brands.append(action[2])

            li.append(len(set(days)))
            li.append(len(set(purchasedays)))
            li.append(len(set(items)))
            li.append(len(set(categories)))
            li.append(len(set(brands)))
            sum_ct = click_ct+cart_ct+purchase_ct+fav_ct
            li.append(float(click_ct)/sum_ct)
            li.append(float(cart_ct)/sum_ct)
            li.append(float(purchase_ct)/sum_ct)
            li.append(float(fav_ct)/sum_ct)

            ds.append(li)
            labels.append(A[i,4])

    ds = np.array(ds,dtype=np.float)
    labels = np.array(labels)

    return ds,labels
