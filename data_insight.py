#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd ,numpy as np,scipy as sp 
from scipy import spatial
from collections import Counter, defaultdict
import csv
import re


# generate negative sample
# device, ip, pv
# cookie, ip, pv
device_train = pd.read_csv('dev_train_basic.csv')
cookie_all_basic = open('cookie_all_basic.csv').readlines()
cookie_mat = []
for line in cookie_all_basic:
	curLine = line.strip().split(',')
	cookie_mat.append(curLine)

cookie_mat = filter(lambda line: line[0] != '-1', cookie_mat[1:])

device = pd.DataFrame(device_train).ix[:,1].unique()
cookie = pd.DataFrame(cookie_mat).ix[:,1].unique()

def device_id_handle(devicefile):
	with open(devicefile) as  fp:
		ValDevHandle = dict()
		fp.readline()
		for line in fp:
			dev = line.split(',')[1]
			handle = line.split(',')[0]
			ValDevHandle[dev] = handle
	return ValDevHandle


def handle_cookie(cookiefile):
	with open(cookiefile) as fp:
		HandleCookie = defaultdict(set)
		fp.readline()
		for line in fp:
			cookie = line.split(',')[1]
			handle = line.split(',')[0]
			HandleCookie[handle].add(cookie)
	return HandleCookie

#############################################version 1 test################################################
## create generate negative sample by selecting the largest Jaccard diatance cookie for the given device ##
###########################################################################################################
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

###########################################################################################################
## create the positive candidate by selecting the same handle cookie and device ,and pair them ############
###########################################################################################################
def generate_positive(ValDevHandle,ValCookieHandle,HandleCookie):
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

# random sample create the negative sample#################################################################
def generate_negative(device, cookie,id_ip,ValDevHandle,ValCookieHandle,length):
	negative_sample = defaultdict(set)
	for i in range(length):
		d = random.choice(device)
		c = random.choice(cookie)
		if (ValDevHandle.get(d, dict().keys()) != ValCookieHandle.get(c, dict().keys())) and ValDevHandle.get(d, dict().keys()) != '-1' and ValCookieHandle.get(d, dict().keys()) != '-1':
			negative_sample[d].add(c)
			print d,c
	return negative_sample

def jaccard_distance(c_media_pv, m_mobile_pv):
	return scipy.spatial.distance.jaccard(c_media_pv,m_mobile_pv)


def load_ip_info(ipfile,XIPS):
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

###########################################################################################################
## calc the mean F05 score of the cross_validation predictions result #####################################
###########################################################################################################
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

###########################################################################################################
## Define the privateness of the IP addresss###############################################################
###########################################################################################################

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

def candidate_generation(device,cookie, id_ip,IPDev,IPCoo):
	Candidates = dict()
	for d in device:
		candidatestotal = set()
		device_ips = id_ip.get(d,dict()).keys()
		for ip in device_ips:
			if(XIPS.get(ip)[0] == 0) and (len(IPDev.get(ip,set()) + IPCoo.get(ip,set()))) <=30:
				candidates = IPCoo.get(ip,dict().keys())
				for candidate in candidates:
					candidatestotal.add(candidate)

		if len(candidatestotal) == 0:
			for ip in device_ips:
				if len(IPDev.get(ip,set()) + IPCoo.get(ip,set())) <=30:
					candidates = IPCoo.get(ip,dict().keys())
					for candidate in candidates:
						candidatestotal.add(candidate)

		if len(candidatestotal) == 0:
			ip_size = dict()
			for ip in device_ips:
				if XIPS.get(ip)[0] ==0:
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

	Candidates[device]=candidatestotal	
	return Candidates

def ip_vector_representation(id):
	alpha = 1.0
	beta = 1.0
	iv_norm = dict()
	iv_sqrt = dict()
	iv_log  = dict()
	for d in id:
		ips = set(id_ip.get(d).keys())
		for ip in ips:
			id_ip_pv_vec_norm.append(id_ip.get(d).get(ip,[0])[0])
			id_ip_pv_vec_sqrt.append((id_ip.get(d).get(ip,[0])[0]+alpha)**0.5)
			id_ip_pv_vec_log.append(np.log(id_ip.get(d).get(ip,[0])[0]+beta))
		sum_norm = np.sum(id_ip_pv_vec_norm)*1.0
		sum_sqrt = np.sum(id_ip_pv_vec_sqrt)*1.0
		sum_log  = np.sum(id_ip_pv_vec_log)*1.0
		iv_norm[d] = 1.0* id_ip_pv_vec_norm/sum_norm
		iv_sqrt[d] = 1.0* id_ip_pv_vec_sqrt/sum_sqrt
		iv_log[d]  = 1.0* id_ip_pv_vec_log /sum_log
	return iv_norm,iv_sqrt, iv_log

def ip_privateness_feature(ips,XIPS,IPCoo, IPDev):
	W_ip_feature = dict()
	for ip in ips:
		temp = []
		temp.append(XIPS.get(ip, dict().keys())[0])
		temp.append(XIPS.get(ip, dict().keys())[1]**(0.5))
		temp.append(XIPS.get(ip, dict().keys())[2])
		temp.append(XIPS.get(ip, dict().keys())[3]**(0.5))
		temp.append(XIPS.get(ip, dict().keys())[4]**(0.5))
		temp.append(len(IPCoo.get(ip,dict()).keys()))
		temp.append(len(IPDev.get(ip,dict()).keys()))
		W_ip_feature[ip] = temp
	return W_ip_feature

def ip_footprint_similarity(device_id,cookie_id,iv):
	device_ip = id_ip.get(device_id,dict()).keys()
	cookie_ip = id_ip.get(cookie_id,dict()).keys()
	intersec_ip = list(set(device_ip) & set(cookie_ip))
	s_sum = s_dot = 0.0
	for ip in intersec_ip:
		s_sum += (ip_vector_representation(cookie)+ip_vector_representation(device_id)).dot(W_ip(ip))
		s_dot += ip_vector_representation(cookie).dot(ip_vector_representation(device_id)).dot(W_ip(ip))
	return s_sum, s_dot

def sigmoid(x): 
    return 1.0/(1+exp(-x)) 

def likelihood(x,y):
	return sigmoid()
















if __main__():
	ValDevHandle = device_id_handle('dev_train_basic.csv')
	ValCookieHandle = device_id_handle('cookie_all_basic.csv')
	HandleCookie = handle_cookie('cookie_all_basic.csv')
	XIPS = loadIPAGG('ipagg_all.csv')
	ips = XIPS.keys()
	id_ip,IPDev, IPCoo = load_ip_info('id_all_ip.csv')
	Candidates = candidate_generation_1(device,cookie, id_ip,IPDev,IPCoo)
	positive_sample,length = generate_positive(ValDevHandle,ValCookieHandle,HandleCookie)
	negative_sample = generate_negative(device, cookie,id_ip,ValDevHandle,ValCookieHandle,length)






