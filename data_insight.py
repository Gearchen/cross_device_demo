#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd ,numpy as np,scipy as sp 
from scipy import spatial
from collections import Counter, defaultdict
import csv
import re

##Handle vs device id,apply for training data
def device_id_handle(devicefile):
	with open(devicefile) as  fp:
		ValDevHandle = defaultdict(set)
		fp.readline()
		for line in fp:
			dev = line.split(',')[1]
			handle = line.split(',')[0]
			ValDevHandle[dev].add(handle)
	return ValDevHandle


##Handle vs cookie id ,apply for training data
def handle_cookie(cookiefile):
	with open(cookiefile) as fp:
		HandleCookie = defaultdict(set)
		fp.readline()
		for line in fp:
			if line.split(',')[0] != '-1':
				cookie = line.split(',')[1]
				handle = line.split(',')[0]
				HandleCookie[handle].add(cookie)
	return HandleCookie

def cookie_handle(cookiefile):
	with open(cookiefile) as fp:
		ValCookieHandle = defaultdict(set)
		fp.readline()
		for line in fp:
			if line.split(',')[0] != '-1':
				cookie = line.split(',')[1]
				handle = line.split(',')[0]
				ValCookieHandle[cookie].add(handle)
	return ValCookieHandle

def all_cookie_basic(cookiefile):
	with open(cookiefile) as fp:
		HandleCookie = defaultdict(set)
		fp.readline()
		for line in fp:
			cookie = line.split(',')[1]
			handle = line.split(',')[0]
			HandleCookie[handle].add(cookie)
	return HandleCookie	

## create generate negative sample by selecting the largest Jaccard diatance cookie for the given device for test
def generate_negative_sample(device, cookie,id_ip,ValDevHandle,ValCookieHandle):
	negative_sample = []
	for d in device:
		max_jacd = -1
		best_candidate = '0'
		for c in cookie:
			# print d,c
			if ValDevHandle.get(d, dict().keys()) != ValCookieHandle.get(c, dict().keys()):
				ip_union = set(id_ip.get(d).keys())| set(id_ip.get(c).keys())
				m_ip_pv_vec = []
				c_ip_pv_vec = []
				for ip in ip_union:
					m_ip_pv_vec.append(id_ip.get(d).get(ip,[0])[0])
					c_ip_pv_vec.append(id_ip.get(c).get(ip,[0])[0])
				temp_jacd = jaccard_distance(m_ip_pv_vec,c_ip_pv_vec)
				# print temp_jacd,m_ip_pv_vec, c_ip_pv_vec
				if temp_jacd > max_jacd:
					max_jacd = temp_jacd
					best_candidate = c
		negative_sample.append([d,c])
	return negative_sample


## create the positive candidate by selecting the same handle cookie and device ,and pair them
def generate_positive(ValDevHandle,HandleCookie,device):
	length = 0
	positive_sample = defaultdict(set)
	for d in device:
		handle_id = ValDevHandle.get(d)
		if handle_id != '-1':
			cookies = HandleCookie.get(handle_id,dict().keys())
			for cookie in cookies:
				positive_sample[d].add(cookie)
				length +=1
	return positive_sample,length

# random sample create the negative sample
def generate_negative(device, cookie,ValDevHandle,ValCookieHandle,length):
	negative_sample = defaultdict(set)
	for i in range(length):
		d = random.choice(device)
		c = random.choice(cookie)
		if (ValDevHandle.get(d, dict().keys()) != ValCookieHandle.get(c, dict().keys())) and (ValDevHandle.get(d, dict().keys()) != '-1') and (ValCookieHandle.get(c, dict().keys()) != '-1'):
			negative_sample[d].add(c)
			# print d,c,ValDevHandle.get(d, dict().keys()), ValCookieHandle.get(c, dict().keys())
	return negative_sample

def jaccard_distance(c_media_pv, m_mobile_pv):
	return scipy.spatial.distance.jaccard(c_media_pv,m_mobile_pv)


def load_ip_info(ipfile):
	IPCoo=defaultdict(set)
	IPDev=defaultdict(set)
	with open(ipfile) as fp:
		fp.readline()
		id_ip  = dict()
		for line in fp:
			matchObj = re.match( r'([a-zA-Z0-9_]*),([0-9\-]*),{([(a-zA-Z0-9(),\-_]*)}', line, flags = 0)
			ips = re.findall(r'(\w*,\w*,\w*,\w*,\w*,\w*,\w*)',matchObj.group(3))
			ValIPS = dict()
			for ip in ips:
				Indiv = ip.split(',')
				arr = np.zeros(6)
				arr[0] = np.float_(Indiv[1])
				arr[1] = np.float_(Indiv[2])
				arr[2] = np.float_(Indiv[3])
				arr[3] = np.float_(Indiv[4])
				arr[4] = np.float_(Indiv[5])
				arr[5] = np.float_(Indiv[6])
				ValIPS[Indiv[0]] = arr

			id = line.split(',')[0]
			id_type = line.split(',')[1]
			id_ip[id] = ValIPS

			if(id_type == '0'):
				for k in ValIPS.keys():
					IPDev[k].add(id)
			else:
				for k in ValIPS.keys():
					IPCoo[k].add(id)
	return id_ip,IPDev, IPCoo


## calc the mean F05 score of the cross_validation predictions result
def calculateF06(results, targets):
	BetaQ = 0.25
	F05 = []
	for i in len(results):
		result = results[i]
		target = targets[i]
		tp = np.float_(len(result & target))
		fp = np.float_(len(result) - tp)
		fn = np.float_(len(target) - tp)
		p = tp/(tp + fp)
		r = tp/(tp + fn)
		f05 = (1.0 + BetaQ)*(p + r)/(BetaQ*p + r)
		F05.appned(f05)
	return np.mean(F05)


## Define the privateness of the IP addresss
def loadIPAGG(ipaggfile):
	XIPS = dict()
	with open(ipaggfile, 'rb') as csvfile:
		reader = csv.reader(csvfile,delimiter = ',')
		reader.next()
		for row in reader:
			datoIP = np.zeros(5)
			datoIP[0] = np.float_(row[1])
			datoIP[1] = np.float_(row[2])
			datoIP[2] = np.float_(row[3])
			datoIP[3] = np.float_(row[4])
			datoIP[4] = np.float_(row[5])
			XIPS[row[0]] = datoIP
	return XIPS

def candidate_generation(device, id_ip,IPDev,IPCoo):
	Candidates = dict()
	for d in device:
		candidatestotal = set()
		device_ips = id_ip.get(d,dict()).keys()
		for ip in device_ips:
			if(XIPS.get(ip)[0] == 0) and  (len(IPDev.get(ip,set())) + len(IPCoo.get(ip,set()))) <30:
				candidates = IPCoo.get(ip,dict().keys())
				for candidate in candidates:
					candidatestotal.add(candidate)

		if len(candidatestotal) == 0:
			for ip in device_ips:
				if len(IPDev.get(ip,set())) + len(IPCoo.get(ip,set())) <30:
					candidates = IPCoo.get(ip,dict().keys())
					for candidate in candidates:
						candidatestotal.add(candidate)

		if len(candidatestotal) == 0:
			ip_size = dict()
			for ip in device_ips:
				if XIPS.get(ip)[0] ==0  and len(IPDev.get(ip,dict().keys())) >0 and len(IPCoo.get(ip,dict().keys()))>0:
					ip_size[ip] = len(IPDev.get(ip,dict().keys())) + len(IPCoo.get(ip,dict().keys()))
			ip_size = sorted(ip_size.items(), lambda x, y: cmp(x[1], y[1]))
			ips = []
			for i in range(min(5, len(ip_size))):
				ip = ip_size[i][0]
				candidates = IPCoo.get(ip,dict().keys())
				for candidate in candidates:
					candidatestotal.add(candidate)

		if len(candidatestotal) == 0:
			ip_size = dict()
			for ip in device_ips:
				ip_size[ip] = len(IPDev.get(ip,dict().keys())) + len(IPCoo.get(ip,dict().keys()))
			ip_size = sorted(ip_size.items(), lambda x, y: cmp(x[1], y[1]))
			ips = []
			for i in range(min(5, len(ip_size))):
				ip  = ip_size[i][0]
				candidates = IPCoo.get(ip,dict().keys())
				for candidate in candidates:
					candidatestotal.add(candidate)	

		Candidates[d]=candidatestotal	
	return Candidates

def ip_norm_vector_representation(id):
	ips = id_ip.get(d).keys()
	id_ip_pv_vec_norm = []
	for ip in ips:
		id_ip_pv_vec_norm.append(id_ip.get(d).get(ip,[0])[0])
	sum_norm = np.sum(id_ip_pv_vec_norm)*1.0
	for i in range(len(id_ip_pv_vec_norm)):
		id_ip_pv_vec_norm[i] = 1.0* id_ip_pv_vec_norm[i]/sum_norm
	return id_ip_pv_vec_norm

def ip_sqrt_vector_representation(id):
	alpha = 1.0
	id_ip_pv_vec_sqrt = []
	ips = id_ip.get(d).keys()
	for ip in ips:
		id_ip_pv_vec_sqrt.append((id_ip.get(d).get(ip,[0])[0]+alpha)**0.5)
	sum_sqrt = np.sum(id_ip_pv_vec_sqrt)*1.0
	for i in range(len(id_ip_pv_vec_sqrt)):
		id_ip_pv_vec_sqrt[i] = 1.0* id_ip_pv_vec_sqrt[i]/sum_norm
	return id_ip_pv_vec_sqrt

def ip_log_vector_representation(id):
	beta = 1.0
	id_ip_pv_vec_log  = []
	ips = id_ip.get(d).keys()
	for ip in ips:
		id_ip_pv_vec_log.append(np.log(id_ip.get(d).get(ip,[0])[0]+beta))
	sum_log  = np.sum(id_ip_pv_vec_log)*1.0
	for i in range(len(id_ip_pv_vec_log)):
		id_ip_pv_vec_log[i] = 1.0* id_ip_pv_vec_log[i]/sum_norm
	return id_ip_pv_vec_log


def ip_privateness_feature(ip,XIPS,IPCoo, IPDev):
	temp = []
	temp.append(1.0)
	temp.append(XIPS.get(ip, dict().keys())[0])
	temp.append(XIPS.get(ip, dict().keys())[1]**(0.5))
	temp.append(XIPS.get(ip, dict().keys())[2])
	temp.append(XIPS.get(ip, dict().keys())[3]**(0.5))
	temp.append(XIPS.get(ip, dict().keys())[4]**(0.5))
	temp.append(len(IPCoo.get(ip,dict().keys())))
	temp.append(len(IPDev.get(ip,dict().keys())))
	return temp

def ip_footprint_similarity_ssum(device_id,cookie_id,weights,XIPS, IPCoo, IPDev):
	device_ip = id_ip.get(device_id,dict()).keys()
	cookie_ip = id_ip.get(cookie_id,dict()).keys()
	intersec_ip = list(set(device_ip) & set(cookie_ip))
	s_sum = 0.0
	for ip in intersec_ip:
		W_ip_feature = ip_privateness_feature(ip, XIPS, IPCoo, IPDev)
		s_sum += (ip_norm_vector_representation(cookie)+ip_norm_vector_representation(device_id)).dot(weights.dot(W_ip_feature))
	return s_sum


def ip_footprint_similarity_sdot(device_id,cookie_id,weights, XIPS, IPCoo, IPDev):
	device_ip = id_ip.get(device_id,dict()).keys()
	cookie_ip = id_ip.get(cookie_id,dict()).keys()
	intersec_ip = list(set(device_ip) & set(cookie_ip))
	s_dot = 0.0
	for ip in intersec_ip:
		s_dot += np.array(ip_norm_vector_representation(cookie)).dot(np.array(ip_norm_vector_representation(device_id)))*(weights.dot(ip_privateness_feature(ip,XIPS,IPCoo, IPDev)))
	return s_dot

def create_trainset(positive_sample, negative_sample):
	train_set = []
	label = []
	for key in positive_sample.keys():
		for cookie in positive_sample.get(key):
			train_set.append([key, cookie])
			label.append(1)
	for key in negative_sample.keys():
		for cookie in negative_sample.get(key):
			train_set.append([key, cookie])
			label.append(0)
	return train_set, label

def sigmoid(x): 
    return 1.0/(1+np.exp(-x)) 

def cost(alpha, device,positive_sample,negative_sample,XIPS,IPCoo,IPDev,weights):
	weights_sum = 0.0
	sum_value = 0.0
	p_cookies = positive_sample.get(device)
	n_cookies = negative_sample.get(device)
	for p in p_cookies:
		for n in n_cookies:
			sum_value += (-np.log(ip_footprint_similarity_sdot(device, p,weights,XIPS,IPCoo,IPDev) - ip_footprint_similarity_sdot(device,n,weights,XIPS,IPCoo,IPDev)))
	for item in weights:
		weights_sum += item**2
	cost_value = sum_value + alpha* weights_sum
	return cost_value

def stochasticGradDesc(devices, positive_sample,negative_sample,XIPS,IPCoo,IPDev):
	alpha  =0.01
	weights = np.zeros(8)
	for device in devices:
		weights += alpha* weights *cost(0.1,device, positive_sample,negative_sample,XIPS,IPCoo,IPDev,weights)
	return weights


def generate_likelihood(deviceTest, Candidates,t,weights, XIPS, IPCoo, IPDev):
	device_cookie_likelihood  = dict()
	for device_id in devices:
		for cookie_id in Candidates.get(device_id):
			device_cookie_likelihood[(device_id,cookie_id)]  = sigmoid(t +ip_footprint_similarity_sdot(device_id,cookie_id,weights, XIPS, IPCoo, IPDev))
	return device_cookie_likelihood






if __main__():
	device_train = pd.read_csv('dev_train_basic.csv')
	device_test = pd.read_csv('dev_test_basic.csv')
	cookie_all_basic = open('cookie_all_basic.csv').readlines()
	cookie_mat = []
	for line in cookie_all_basic:
		curLine = line.strip().split(',')
		cookie_mat.append(curLine)

	cookieAll =  pd.DataFrame(cookie_mat).ix[:,1].unique()
	cookie_mat = filter(lambda line: line[0] != '-1', cookie_mat[1:])

	cookieKnown = pd.DataFrame(cookie_mat).ix[:,1].unique()
	device = device_train.ix[:,1].unique()
	deviceTest = device_test.ix[:,1].unique()
	ValDevHandle = device_id_handle('dev_train_basic.csv') # device with Handle
	ValcookieHandle = cookie_handle('cookie_all_basic.csv') # coookie_handle 
	HandleCookie = handle_cookie('cookie_all_basic.csv') # cookie with Known Hanldle
	AllCookie = all_cookie_basic('cookie_all_basic.csv')
	XIPS = loadIPAGG('ipagg_all.csv')
	ips = XIPS.keys()
	id_ip,IPDev, IPCoo = load_ip_info('id_all_ip.csv')
	# select candidate 
	W_ip_feature = ip_privateness_feature(ips,XIPS,IPCoo, IPDev)
	Candidates = candidate_generation(deviceTest,id_ip,IPDev,IPCoo)
	positive_sample,length = generate_positive(ValDevHandle,HandleCookie,device)
	negative_sample = generate_negative(device, cookieKnown,ValDevHandle,ValCookieHandle,length)
	train_set, label = create_trainset(positive_sample, negative_sample)
	weights = stochasticGradDesc(devices, positive_sample,negative_sample,XIPS,IPCoo,IPDev)

	

def generate_device_user_likelihood(Candidates,ValcookieHandle,device_cookie_likelihood):
	generate_device_user_likelihood = dict()
	device_user = defaultdict(set)
	for device_id in deviceTest:
		cookies_temp = Candidates.get(device_id)
		user_likelihood = defaultdict(set)
		for cookie_id in cookies_temp:
			user = ValcookieHandle.get(cookie_id)
			user_likelihood[user].add(device_cookie_likelihood.get((device_id, cookie_id)))
		for user in user_likelihood.keys():
			generate_device_user_likelihood[(device, user)] = max(user_likelihood.get(user))
			device_user[device_id] = user
	return generate_device_user_likelihood,device_user

def SmartGen(device_id, k,device_user,generate_device_user_likelihood):
	user_candidate = []
	precision = []
	candidates_n = estimation = candidates_l = 0.0
	users = device_user.get(device_id)
	likelihood =estimation_list =  []
	for user in users:
		likelihood.append(generate_device_user_likelihood.get(user))
	for k in range(1, len(users)):
		for i in ~np.argsort(likelihood)[:k]:
			user = users[i]
			user_candidate.append(user)
			candidates_n += len(AllCookie.get(user))
			candidates_l += generate_device_user_likelihood.get(user)
		for i in ~np.argsort(likelihood)[:k]:
			user = users[i]
			precision = np.float_(len(AllCookie.get(user))) / np.float_(candidates_n)
			estimation += (generate_device_user_likelihood.get(user)*1.0 / candidates_l)* ((1.25*precision) / (0.25*precision + 1.0))
		estimation_list.append(estimation)
	k_final = ~np.argsort(estimation_list)
	cookie_final = []
	for i in ~np.argsort(likelihood)[:k_final]:
		user = users[i]
		cookie_candidates = HandleCookie.get(user)
		for cookie in cookie_candidates:
			cookie_final.append(cookie)
	return cookie_final

def SmartGen+(device_id, k,device_user,generate_device_user_likelihood):
	user_candidate = []
	precision = []
	candidates_n = estimation = candidates_l = 0.0
	users = device_user.get(device_id)
	likelihood = []
	for user in users:
		likelihood.append(generate_device_user_likelihood.get(user))
	user = ~np.argsort(likelihood)[0]




















