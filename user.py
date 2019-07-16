import numpy as np
#用户特征初始化
def feature_ini():	
	gender=[]
	age=[]
	profession=[]
	share=[]
	locationdis=[]
	locationdir=[]
	usefeature=[]
	gender=np.random.random_integers(0,1,500)
	age=np.random.random_integers(1,10,500)
	profession=np.random.random_integers(1,10,500)
	share=np.random.uniform(0,1,500)
	locationdir=np.random.random_integers(0,9,500)
	locationdis=np.random.random_integers(0,9,500)
	usefeature=np.c_[gender,age,profession,share,locationdis,locationdir]
	#print (share)
	#np.savetxt('feature.csv',usefeatur,delimiter=',')
#用户特征更改
def  ueser_feature_change():#每天更改
	pass
	#share_1=0.5*share+0.5*

#用户请求
def user_request(request_tot):
	Request=np.random.poisson(lam=3.0,size=None)
	user_Request_pro=Request/5
	cont=[]
	content_id=[]
	request_id=[]
	t=0
	user_Request_history={}

	for i in range(0,5):
		t+=1
		Requestis=np.random.uniform(0,1)
		#print(Request,Requestis,user_Request_pro)
		if user_Request_pro>Requestis:
			user_Request_is=1
			req_id=np.random.zipf(2.0,)
			content_id.append(req_id)
		else:
				user_Request_is=0
				content_id.append(0)
				pass
		cont.append(user_Request_is)
	
	request=np.argsort(-np.asarray(request_tot))
	request=request.tolist()
	for j in content_id:
		if j-1<0 or j>20:
			request_id.append('None')
			continue
		request_id.append(request.index(j-1))

	#print (request_id)


	#content_history=np.loadtxt('history.csv',delimiter=',')
	#content_history=content_history[:,None]
	#print (content_history.shape)
	#n=np.column_stack((content_history,content_id))
	#print (n.shape)
	#np.savetxt('history.csv',n,delimiter=',')
	return request_id
#划分社区
def commu():
	pass
#--------------------------------------------------------------
#----------------------------test------------------------------
#--------------------------------------------------------------
#user_request([12,43,45,51,3,1,3,2,3,5,35,7,6,2,64,7,86,67,24,6625,45,2452,6,623,62,6246,264])
def user_Request_history():
	pass















