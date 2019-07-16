import numpy as np
from math import exp
 

def initialize(u,c,l,w,R,pu_n_1,pu_n,pl_n_1,pl_n,pw_n_1,pw_n,pc_n_1,pc_n):  
    for j in range(5):
        u[j]=np.random.random((20,20))
        l[j]=np.random.random((20,20))
        w[j]=np.random.random((20,20))
        pu_n_1[j]=np.zeros((20,20))
        pu_n[j]=np.zeros((20,20))
        pl_n_1[j]=np.zeros((20,20))
        pl_n[j]=np.zeros((20,20))
        pw_n_1[j]=np.zeros((20,20))
        pw_n[j]=np.zeros((20,20))
    for q in range(20):
        c[q]=np.random.random((20,20))
        pc_n_1[q]=np.zeros((20,20))
        pc_n[q]=np.zeros((20,20))
def g(x):
    g=1/(1+np.exp(-x))      
    return g

def gd(x):
    gd=(np.exp(-x)/pow((1+np.exp(-x)),2))
    return gd    
def update(beta1,beta2,u,c,l,w,pu_n,pl_n,pw_n,pc_n,pu_n_1,pl_n_1,pw_n_1,pc_n_1):#inpot user_id ,content_id
    for user_id in range(5):
        u[user_id]-=(beta1*pu_n_1[user_id]+beta2*pu_n[user_id])
        l[user_id]-=(beta1*pl_n_1[user_id]+beta2*pl_n[user_id])
        w[user_id]-=(beta1*pw_n_1[user_id]+beta2*pw_n[user_id])
        pu_n_1[user_id]=pu_n[user_id]
        pl_n_1[user_id]=pl_n[user_id]
        pw_n_1[user_id]=pw_n[user_id]
    for content_id in range(20):
        c[content_id]-=(beta1*pc_n_1[content_id]+beta2*pc_n[content_id])
        pc_n_1[content_id]=pc_n[content_id]
    #print(u[1],c[1])
def ploss(S,R,u,c,l,w,user_request_history,content_requested_history,pu_n,pl_n,pw_n,pc_n):#所有的 user和conten求导

    for user_id in range(5):
        pua=0
        pla=0
        pwa=0
        for k in user_request_history[user_id]:
            pua+=np.dot(gd(np.linalg.det(np.dot(u[user_id],c[k])))*(g(np.linalg.det(np.dot(u[user_id],c[k])))-R[user_id][k]),c[k])
        pu_n[user_id]=pua+np.subtract(u[user_id],l[user_id])+np.subtract(u[user_id],w[user_id])+u[user_id]*len(user_request_history[user_id])
        #print(pu_n[user_id])
        for qi in range(5):
            if qi==user_id:
                continue
            pla+=np.dot(gd(np.linalg.det(np.dot(l[user_id],w[qi])))*(g(np.linalg.det(np.dot(l[user_id],w[qi])))-S[user_id][qi]),w[qi])
        pl_n[user_id]=pla+(u[user_id]-l[user_id])+19*l[user_id]
        
        for qj in range(5):
            if qj==user_id:
                continue
            pwa+=np.dot(gd(np.linalg.det(np.dot(l[qj],w[user_id])))*(g(np.linalg.det(np.dot(l[qj],w[user_id])))-S[qj][user_id]),l[qj])
        pw_n[user_id]=pwa-(u[user_id]-w[user_id])+19*w[user_id]

    for content_id in range(20):
        pca=0
        for i in content_requested_history[content_id]:
            pca+=np.dot(gd(np.linalg.det(np.dot(u[i],c[content_id])))*(g(np.linalg.det(np.dot(u[i],c[content_id])))-R[i][content_id]),u[i])
        #print (pca)
        pc_n[content_id]=pca+c[content_id]*len(content_requested_history[content_id])
def loss(S,R,u,c,l,w,content_requested_history):
    a=b=c1=d=e=f=g1=h=0.0
    for i in range(5):
        for k in range(20):
            a+=pow(R[i][k]-g(np.linalg.det(np.dot(u[i],c[k]))),2)
        b+=pow(np.linalg.norm(np.subtract(u[i],l[i]),ord=None),2)
        d+=pow(np.linalg.norm(np.subtract(u[i],w[i]),ord='fro'),2)
        e+=pow(np.linalg.norm(u[i],ord=None),2)
        g1+=pow(np.linalg.norm(l[i],ord=None),2)
        h+=pow(np.linalg.norm(w[i],ord=None),2)   
        for j in range(5):
            c1=pow(S[i][j]-g(np.linalg.det(np.dot(l[i],w[j]))),2)
    for k in range(20):
        f+=len(content_requested_history[k])*pow(np.linalg.norm(c[k],ord=None),2)
    #print(a,b,c1,d,e,f,g1,h)
    loss=0.5*(a+b+c1+d+e+f+g1+h)
    return loss
def value2(u,c):
    value2_list_tmp=np.zeros((5,20))
    value2_list_tmp=value2_list_tmp.tolist()
    value2_list=[]
    for user_id in range(5):
        for content_id in range(20):
            value2=np.linalg.det(np.dot(u[user_id],c[content_id])) 
            value2=g(value2)
            value2_list_tmp[user_id][content_id]=float(value2)
    for k in range(20):
        value2_ave_tmp=0
        for i in range(5):
            value2_ave_tmp+=value2_list_tmp[i][k]
            value2_ave=value2_ave_tmp/5
        value2_list.append(value2_ave)
    return value2_list
        
#------------------------------------------------
#            测试
#------------------------------------------------
#print("{:.8}".format(gd(1)))
def main():
    initialize()
    loss_tmp=loss()
    ploss()
    update()
    loss_now=loss()
    while(loss_now<loss_tmp):  
        ploss()
        update()
        print(loss())
        loss_tmp=loss_now
        loss_now=loss()
    for  m in range(5):
        for n in range(20):
            print(value2(m,n),m,n)

if __name__=='__main__':
    print('begin',g(0.24))
    main()
    print('end')
