import numpy as np 
import user
#import contentfeature
from Basic_UCB import LinUCB
import PMF
#import tensorflow as tf
#import matplotlib.pyplot as plt
userfeature = {}
xa=np.zeros((5,20))
xa=xa.tolist()
article=[]
user_feature_h={
    0:[0.67,0.66,0.55,0.96,0.44,0.63 ],
    1:[0.85 ,0.04 ,   0.20  ,  0.02   , 0.32   , 0.53 ],
    2:[0.87   ,0.18  ,  0.91   , 0.53 ,   0.22    ,0.94 ],
    3:[0.97   , 0.36  ,  0.12   , 0.14 ,   0.07  ,  0.25 ],
    4:[0.72   , 0.97   , 0.63  ,  0.24 ,   0.67   , 0.64 ],
}
user_feature_y={
    0:[0.00   , 1.00   , 9.00 ,   0.91   , 5.00    ,8.00 ],
    1:[1.00   , 3.00   , 6.00    ,0.38   , 4.00   , 4.00 ],
    2:[1.00  ,  9.00   , 6.00   , 0.44   , 8.00   , 1.00 ],
    3:[0.00   , 9.00   , 6.00   , 0.85   , 4.00  ,  2.00 ],
    4:[1.00  , 9.00   , 7.00  ,  0.40   , 6.00  ,  4.00 ] ,
}
content_feature={
    0:[2  ,0.06    ,1 ,  0   ,36  ,1],
    1:[2   ,0.62    ,1  , 0  , 14 , 1],
    2:[5   ,0.18   , 1   ,0   ,13 , 1],
    3:[5   ,0.37   , 1  , 0  , 3  , 1],
    4:[5   ,0.06   , 1  , 0  , 8  , 1],
    5:[1   ,0.16   , 1   ,0  , 18 , 1],
    6:[5   ,0.26  ,  1   ,0  , 16 ,1],
    7:[2   ,0.39  ,  1   ,0  , 3 ,  1],
    8:[1   ,0.66  ,  1  , 0 ,  0  , 1],
    9:[1   ,0.39   , 1  , 0  , 23 , 1],
    10:[5  , 0.43   , 1   ,0  , 15 , 1],
    11:[5   ,0.88   , 1   ,0  , 2  , 1],
    12:[2  , 0.64   , 1  , 0  , 3  , 1],
    13:[8  , 0.44  ,  1   ,0 ,  2  , 1],
    14:[1  , 0.98   , 1  , 0  , 3  , 1],
    15:[6  , 0.64   , 1   ,0  , 10  ,1],
    16:[4  , 0.70   , 1  , 0  , 12 , 1],
    17:[8  , 0.70   , 1 ,  0 ,  12 , 1],
    18:[5 , 0.95   , 1   ,0 ,  4  , 1],
    19:[5 , 0.86  ,  1  , 0 ,  15 , 1],
    20:[1 ,  0.29  ,  1,   0 , 1  , 1], 
}
u={}#用户特征字典,20*20
c={}#20*20
l={}#20*20
w={}#20*20
pu_n_1={}
pu_n={}
pl_n_1={}
pl_n={}
pw_n_1={}
pw_n={}
pc_n_1={}        
pc_n={}
value_list=[]
beta1=0.02
beta2=0.002


user_request_history={
    0:[0,3,5,7,8,9,16],
    1:[0,4,5,7],
    2:[0,4,7,8,9,16],
    3:[0,6,15,16,17,19],
    4:[0,4,7,15,16,17],}
content_requested_history={
    0:[0,1,2,3,4],
    1:[2],
    2:[4],
    3:[0],
    4:[0,1],
    5:[0,2],
    6:[4],
    7:[2],
    8:[1],
    9:[3],
    10:[0,2],
    11:[4],
    12:[1,4],
    13:[1],
    14:[3,1],
    15:[0],
    16:[0],
    17:[1],
    18:[],
    19:[],}
R=[[0.6791835361541675, 0.5620646467026921, 0.5620882821396256, 0.561937000169359, 0.5620360462552796, 0.5620806684472756, 0.5621063807430544, 0.5616948420717992, 0.5610653513845221, 0.5621116942061967, 0.5621026535783236, 0.5619299250594395, 0.5617303314287778, 0.5620218591710394, 0.5616637946909947, 0.5620824682356382, 0.5620750054776557, 0.56211179666072, 0.561988454836639, 0.5621059473685583], 
[0.9918461643273087, 0.5611157635971838, 0.5615565499715366, 0.5607588709728126, 0.5612753806200942, 0.5612290048224895, 0.5616663793853318, 0.558186241299787, 0.5491151826110179, 0.5615311232215855, 0.5616340747664113, 0.5606115474016872, 0.5582003172848591, 0.5614089325518099, 0.5557093291434417, 0.5615442946562844, 0.5613714207666133, 0.5617728157613046, 0.5609009456446545, 0.5616358440163136], 
[0.9999979799527571, 0.5620832729894395, 0.5621222744379069, 0.5620872592351419, 0.5621075905630621, 0.5620813170832679, 0.5621288062732458, 0.5619591703077076, 0.5617126631753652, 0.5621054790261273, 0.562127108951378, 0.5620841282718699, 0.561963314732529, 0.5621248396237338, 0.5618627947028252, 0.5621253056881046, 0.5621094795878969, 0.5621411028087864, 0.562094484820599, 0.5621278150207608], 
[0.6791150968227563, 0.5612923724823876, 0.5618842027622768, 0.5618193467821768, 0.5618452651638967, 0.5606574905521009, 0.5619047295412212, 0.5606988274515242, 0.5584924007054154, 0.5610391806926769, 0.5619040663617153, 0.5618334961940418, 0.5607879039343965, 0.562016576173067, 0.5594043762811587, 0.5619537882398428, 0.5617863980569299, 0.5620412388210602, 0.5618508571825059, 0.5619163955142817], 
[0.9999996531728194, 0.5620253697479651, 0.5620447969827345, 0.5617210827173822, 0.5619462472029846, 0.5620564765343272, 0.5620758020813631, 0.5613145992962729, 0.559467290834839, 0.5620971410283501, 0.5620694338634007, 0.5616990645459242, 0.5613872961932672, 0.5618624039313553, 0.5613082856863195, 0.5620275179469106, 0.5620273607421059, 0.5620742708292576, 0.5618410219765961, 0.5620747267063636]]
S=[[0.23,0.34,0.34,0.45,0.53],[0.64,0.4,0.72,0.83,0.64],[0.42,0.31,0.73,0.12,0.34],[0.43,0.24,0.23,0.13,0.13],[0.34,0.64,0.88,0.45,0.41]]

request_id=[]
cache_list=[]
for i in range(5):#xa
    for j in range(20):
        user1=np.asarray(user_feature_y.get(i)).reshape(6,1)
        user2=np.asarray(user_feature_h.get(i)).reshape(1,6)
        user_feature=np.dot(user1,user2)
        xa[i][j]=np.dot(user_feature,content_feature.get(j))
for k in range (20):
    article.append(k)


request_tot=[36,14,13,3,8,18,16,3,0,23,15,2,3,2,3,10,12,12,4,15,1]

def value():
    ucb=LinUCB()
    ucb.set_articles(article,xa)
    #value1=ucb.value1(xa,article)
    #print(value1)
    #print(value1.index(max(value1)),max(value1))
    global request_id
    request_id=user.user_request(request_tot)
    ucb.update(request_id)
    value1=list(ucb.value1(xa,article))
    value1=value1[0]
    #print(value1.index(max(value1)),max(value1))
    print ('UCB END and PMF Start')


    PMF.initialize(u,c,l,w,R,pu_n_1,pu_n,pl_n_1,pl_n,pw_n_1,pw_n,pc_n_1,pc_n)
    loss_tmp=PMF.loss(S,R,u,c,l,w,content_requested_history)
    PMF.ploss(S,R,u,c,l,w,user_request_history,content_requested_history,pu_n,pl_n,pw_n,pc_n)
    PMF.update(beta1,beta2,u,c,l,w,pu_n,pl_n,pw_n,pc_n,pu_n_1,pl_n_1,pw_n_1,pc_n_1)
    print (' PMF Update Success')
    loss_now=PMF.loss(S,R,u,c,l,w,content_requested_history)
    while(loss_now<loss_tmp):  
            PMF.ploss(S,R,u,c,l,w,user_request_history,content_requested_history,pu_n,pl_n,pw_n,pc_n)
            PMF.update(beta1,beta2,u,c,l,w,pu_n,pl_n,pw_n,pc_n,pu_n_1,pl_n_1,pw_n_1,pc_n_1)
            #print(PMF.loss(R,u,c,l,w,content_requested_history))
            loss_tmp=loss_now
            loss_now=PMF.loss(S,R,u,c,l,w,content_requested_history)
    value2=PMF.value2(u,c)
    print ('PMF End')
    #print(value1,value2)
    for m in range(len(value1)):
        value=value1[m]+value2[m]
        value_list.append(value)
    #print(value_list)
    print ('Value End')
    return (value_list)
    
def cache(value_list):
    cache_list=[]
    #print (value_list)
    value_select=value_list.copy()
    value_select.sort(key=None,reverse=True)
    for i in range(5):
        #print(i,value_list.index(value_select[i]),request_tot[value_list.index(value_select[i])])
        cache_list.append(value_list.index(value_select[i]))
    print(cache_list)
    return cache_list
def hit_rate():
    hit_cont_no=0  
    hit=0
    for i in request_id:   
        if i is 'None':
            hit_cont_no_=1
        if i in cache_list:
            hit+=1
    try:
        hit_rate=hit/(hit_cont_no+hit)
        
    except:
        hit_rate=0  
    return (hit_rate)
def main():#求值-缓存-请求-hit_rate-更新tot。
    global cache_list
    global value_list
    for m in  range(100):
        value_list.clear()
        value_list=value()
        cache_list=cache(value_list)
        print (request_id)
        print(hit_rate(),m)
    
print('begin',)
main()
print('end')
#-----------------











