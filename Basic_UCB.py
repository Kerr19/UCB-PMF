import numpy as np
class LinUCB(object):
    def __init__(self,):
        self.alpha = 0.25
        self.r1 = 0.5
        self.r0 = -0.02
        self.d = 6  # dimension of user features
        self.Aa = {}# Aa : collection of matrix to compute disjoint part for each article a, d*d
        self.AaI = {}# AaI : store the inverse of all Aa matrix

        self.ba = {}  # ba : collection of vectors to compute disjoin part, d*1
        self.theta = {}

        self.a_max = 0
    def set_articles(self,art,x):
        for key in art:
            self.Aa[key] = np.identity(self.d) # 创建单位矩阵
            self.ba[key] = np.zeros((self.d,1))
            self.AaI[key] = np.identity(self.d)
            self.theta[key] = np.zeros((self.d,1))
        
        self.xT=np.zeros((5,20))
        self.xT=self.xT.tolist()
        self.x=np.zeros((5,20))
        self.x=self.x.tolist()
        for i in range(5):
            for j in range(20):
                self.x[i][j]=np.asarray(x[i][j]).reshape(6,1)
                self.xT[i][j]=np.asarray(self.x[i][j]).reshape(1,6)
                #print (self.x[i][j],self.xT[i][j])
    def g(self,x):
        g=1/(1+np.exp(-x))      
        return g           


    def update(self,request_id):
        for user_id in range(len(request_id)):
            if request_id[user_id] !="None" and request_id[user_id]<20:
                r = self.r1
                content_id=request_id[user_id]
                self.Aa[content_id] += np.dot(self.x[user_id][content_id],self.xT[user_id][content_id])
                self.ba[content_id] += r * (self.x[user_id][content_id])
                self.AaI[content_id] = np.linalg.inv(self.Aa[content_id])
                self.theta[content_id] = np.dot(self.AaI[content_id],self.ba[content_id])
        for k in range(20):
            if k in request_id:
                continue
            elif k not in request_id:
                r = self.r0
                content_id=k
                for userid in range(len(request_id)):
                    self.Aa[content_id] += np.dot(self.x[userid][content_id],self.xT[userid][content_id])
                    self.ba[content_id] += r * (self.x[userid][content_id])
                    self.AaI[content_id] = np.linalg.inv(self.Aa[content_id])
                    self.theta[content_id] = np.dot(self.AaI[content_id],self.ba[content_id])
                
        print('UCB update success')

    def value1(self,features,articles):
        value1_list_tmp=np.zeros((5,20))
        value1_list_tmp=value1_list_tmp.tolist()
        value1_list=[]
        for i in range(5):
            for article in articles:

                user_cont=features[i][article]
                #print (user_cont)
                xaT = np.array([user_cont]) # d * 1
                xa = np.transpose(xaT)

                AaI_tmp = np.array(self.AaI[article])
                theta_tmp = np.array(self.theta[article])
                #art_max = articles[np.argmax(np.dot(xaT,theta_tmp) + self.alpha * np.sqrt(np.dot(np.dot(xaT,AaI_tmp),xa)))]
                value=self.g(np.dot(xaT,theta_tmp) + self.alpha * np.sqrt(np.dot(np.dot(xaT,AaI_tmp),xa)))
                #print(value.tolist())
                value1_list_tmp[i][article]=float(value)
        #print(value1_list_tmp)
        for k in articles:
            value_ave=0
            for i in range(5):
                value_ave+=value1_list_tmp[i][k]
                value_ave=value_ave/5
            value1_list.append(value_ave)
        #print(type(value1_list)      )
        #self.a_max = art_max

        #return self.a_max,
        return value1_list,
#---------------------------------------------------------------
#
#                             测试                              
#
#---------------------------------------------------------------
#ucb=LinUCB()
#article=[0,1,2,3,4,5]
#request_id=['None', 'None', 2, 1, 3]
#ucb.set_articles(article,)
#print(ucb.Aa[0])
#ucb.value1(user,content_feature,article)
#ucb.update(request_id)