import pandas as pd ,numpy as np,scipy as sp 
from scipy import spatial
from collections import Counter, defaultdict
import csv
import re
import random
import ua_parser
from ua_parser import user_agent_parser
import pprint
import xgboost as xgb
import sklearn 
from sklearn import cross_validation
from sklearn.ensemble.forest import RandomForestRegressor

import multiprocessing

def id_handle(idfile):
	with open(idfile) as fp:
		ValCookieHandle = defaultdict(set)
		ValDevHandle = defaultdict(set)
		fp.readline()
		for line in fp:
			handle = line.strip().split(',')[0]
			cookie = line.strip().split(',')[1]
			device = line.strip().split(',')[2]
			ValCookieHandle[cookie].add(handle)
			ValDevHandle[device].add(handle)
	return ValCookieHandle, ValDevHandle

def handle_id(idfile):
	with open(idfile) as fp:
		ValHandleCookie = defaultdict(set)
		ValHandleDevice = defaultdict(set)
		fp.readline()
		for line in fp:
			handle = line.strip().split(',')[0]
			cookie = line.strip().split(',')[1]
			device = line.strip().split(',')[2]
			ValHandleCookie[handle].add(cookie)
			ValHandleDevice[handle].add(device)
	return ValHandleCookie, ValHandleDevice

def device_ip_list(id_list, ipfile):
	ValDeviceIP = defaultdict(set)
	ValIPDevice = defaultdict(set)
	with open(ipfile) as fp:
		for line in fp:
			device = line.strip().split(',')[0]
			ip = line.strip().split(',')[-9]
			ValDeviceIP[device].add(ip)
			ValIPDevice[ip].add(device)
	return ValDeviceIP, ValIPDevice

def cookie_ip_list(id_list, ipfile):
	ValCookieIP = defaultdict(set)
	ValIPCookie = defaultdict(set)
	with open(ipfile) as fp:
		for line in fp:
			cookie = line.strip().split(',')[0]
			ip = line.strip().split(',')[-7]
			ValCookieIP[cookie].add(ip)
			ValIPCookie[ip].add(cookie)
	return ValCookieIP, ValIPCookie

def device_pro_list(ipfile):
	ValDevicePro = defaultdict(set)
	with open(ipfile) as fp:
		for line in fp:
			device = line.strip().split(',')[0]
			properties = line.strip().split(',')[-2]
			ValDevicePro[device].add(properties)
	return ValDevicePro

def cookie_pro_list(ipfile):
	ValCookiePro = defaultdict(set)
	with open(ipfile) as csvfile:
		spamreader = csv.reader(csvfile, delimiter = ',')
		for line in spamreader:
			cookie = line[0]
			properties = line[-2]
			ValCookiePro[cookie].add(properties)
	return ValCookiePro

def parser_device_ua():
	device_brand  = defaultdict(set)
	device_family = defaultdict(set)
	device_model  = defaultdict(set)
	os_family = defaultdict(set)
	user_agent_family = defaultdict(set)
	with open('CrossWise_deviceid_result.csv','rb') as csvfile:
		spamreader = csv.reader(csvfile, delimiter =',')
		for row in spamreader:
			device = row[0]
			parsed_string = user_agent_parser.Parse(row[3])
			device_brand[device].add(parsed_string.get('device').get('brand'))
			device_family[device].add(parsed_string.get('device').get('family'))
			device_model[device].add(parsed_string.get('device').get('model'))
			os_family[device].add(parsed_string.get('os').get('family'))
			user_agent_family[device].add(parsed_string.get('user_agent').get('family'))
	return device_brand, device_family,device_model, os_family,user_agent_family


def parser_cookie_ua():
	cookie_brand  = defaultdict(set)
	cookie_family = defaultdict(set)
	cookie_model  = defaultdict(set)
	os_family = defaultdict(set)
	user_agent_family = defaultdict(set)
	with open('CrossWise_cookie.csv','rb') as csvfile:
		spamreader = csv.reader(csvfile, delimiter =',')
		for row in spamreader:
			cookie = row[0]
			parsed_string = user_agent_parser.Parse(row[2])
			cookie_brand[device].add(parsed_string.get('device').get('brand'))
			cookie_family[device].add(parsed_string.get('device').get('family'))
			cookie_model[device].add(parsed_string.get('device').get('model'))
			os_family[device].add(parsed_string.get('os').get('family'))
			user_agent_family[device].add(parsed_string.get('user_agent').get('family'))
	return cookie_brand, cookie_family,cookie_model, os_family,user_agent_family	

# test the number cross ip and not belong to one person
def create_negative_sample_rule_log(Devices, Cookies, ValCookieIP, ValDeviceIP, ValDevHandle, ValCookieHandle, alpha):
	ip_cross_negative_rule = defaultdict(set)
	for device in Devices:
		cookie_negative = []
		jaccard_distance = []
		for cookie in Cookies:
			device_ip = ValDeviceIP.get(device)
			cookie_ip = ValCookieIP.get(cookie)
			if len(device_ip & cookie_ip) >0 and (len(ValDevHandle.get(device) & ValCookieHandle.get(cookie)) < 1):
				cookie_negative.append(cookie)
		if len(cookie_negative) >0:
			for cookie in cookie_negative:
					jaccard_distance.append(np.log(1.0*len(ValDevicePro.get(device) & ValCookiePro.get(cookie)) + alpha) / np.log(1.0*len(ValDevicePro.get(device) |ValCookiePro.get(cookie)) + alpha))
			cookie_final = cookie_negative[~np.argsort(jaccard_distance)[0]]
			ip_cross_negative_rule[device].add(cookie_final)
			print device,cookie_final, jaccard_distance[~np.argsort(jaccard_distance)[0]]
	return ip_cross_negative_rule

def create_negative_sample_rule_sqrt(Devices, Cookies, ValCookieIP, ValDeviceIP, ValDevHandle, ValCookieHandle, alpha):
	ip_cross_negative_rule = defaultdict(set)
	for device in Devices:
		cookie_negative = []
		jaccard_distance = []
		for cookie in Cookies:
			device_ip = ValDeviceIP.get(device)
			cookie_ip = ValCookieIP.get(cookie)
			if len(device_ip & cookie_ip) >0 and (len(ValDevHandle.get(device) & ValCookieHandle.get(cookie)) < 1):
				cookie_negative.append(cookie)
		if len(cookie_negative) >0:
			for cookie in cookie_negative:
					jaccard_distance.append(np.sqrt(1.0*len(ValDevicePro.get(device) & ValCookiePro.get(cookie)) + alpha) / np.sqrt(1.0*len(ValDevicePro.get(device) | ValCookiePro.get(cookie)) + alpha))
			cookie_final = cookie_negative[~np.argsort(jaccard_distance)[0]]
			ip_cross_negative_rule[device].add(cookie_final)
			print device,cookie_final, jaccard_distance[~np.argsort(jaccard_distance)[0]]
	return ip_cross_negative_rule


def create_negative_sample_random(Devices, Cookies, ValCookieIP, ValDeviceIP, ValDevHandle, ValCookieHandle):
	ip_cross_negative_random = defaultdict(set)
	for device in Devices:
		cookie_negative = []
		for cookie in Cookies:
			device_ip = ValDeviceIP.get(device)
			cookie_ip = ValCookieIP.get(cookie)
			if len(device_ip & cookie_ip) >0 and (len(ValDevHandle.get(device) & ValCookieHandle.get(cookie, set())) < 1):
				ip_cross_negative_random[device].add(cookie)
	return ip_cross_negative_random

def create_positive_sample(idfile):
	ip_cross_positive_sample = defaultdict(set)
	with open(idfile,'rb') as csvfile:
		spamreader = csv.reader(csvfile, delimiter =',')
		for line in spamreader:
			handle_id = line[0]
			cookie = line[1]
			device = line[2]
			ip_cross_positive_sample[device].add(cookie)
	return ip_cross_positive_sample

def get_device_view(idfile):
	ValDeviceView = defaultdict(int)
	with open(idfile,'rb') as csvfile:
		spamreader = csv.reader(csvfile, delimiter = ',')
		for line in spamreader:
			device = line[0]
			pv = int(line[11])
			ValDeviceView[device] += pv
	return ValDeviceView

def get_cookie_view(idfile):
	ValCookieView = defaultdict(int)
	with open(idfile,'rb') as csvfile:
		spamreader = csv.reader(csvfile, delimiter = ',')
		for line in spamreader:
			cookie= line[0]
			ValCookieView[cookie] +=1
	return ValCookieView

def property_type_dict():
	property_type = defaultdict(set)
	with open('media_feature_new.csv','rb') as csvfile:
		spamreader = csv.reader(csvfile, delimiter = ',')
		for line in spamreader:
			media = line[0]
			properties_1 = line[9]
			properties_2 = line[-1]
			if properties_1 != '':
				property_type[media].add(properties_1)
			if properties_2 != '':
				property_type[media].add(properties_2)
		return property_type

# ip vector : is_cellu, total view, sqrt of pc os, sqrt of mobile model, number of cookies, number of mobiles
'''
def ip_feature_vector(ips):
	ip_feature = np.zeros(6)
	niprows=0
	for ip in ips:
		ip_feature_vector = []
		ip_feature_vector.append(ip)
		ip_feature_vector.append(ValIPCookieView.get(ip)+ValIPDeviceView.get(ip))
		ip_feature_vector.append(np.sqrt(len(ValIPOS.get(ip))))
		ip_feature_vector.append(np.sqrt(len(ValIPModel.get(ip))))
		ip_feature_vector.append(len(ValIPCookie.get(ip,dict()).keys()))
		ip_feature_vector.append(len(ValIPDevice.get(ip,dict()).keys()))
		ip_feature += ip_feature_vector
		niprows +=1
	return np.array(ip_feature), niprows
'''

def create_dataset(Candidates, ValDeviceIP, ValDevicePro, ValCookieIP, ValCookiePro, ValIPDevice, ValIPCookie, ValCookieView, ValIPDeviceView, alpha):
	numpatterns = 0
	for k,v in Candidates.iteritems():
		numpatterns = numpatterns + len(v)

	dataSet = []
	Added = 0
	Y = []

	for k,v in Candidates.iteritems():
		device = k
		setdevips = set(ValDeviceIP.get(device, set()))
		setdevpros= set(ValDevicePro.get(device,set()))

		for cookie in v:
			row = []

			setcooips = set(ValCookieIP.get(cookie,dict().keys()))
			setcoopros = set(ValCookiePro.get(cookie, dict().keys()))

			setcooprostypes = set()
			setdevprostypes = set()

			for p in setdevpros:
				p_types = property_type.get(p, set())
				for t in p_types:
					setdevprostypes.add(t)

			for p in setcoopros:
				p_types = property_type.get(p, set())
				for t in p_types:
					setcooprostypes.add(t)

			ProTypes = setdevprostypes & setcooprostypes
			ProTypes_union = setdevprostypes | setcooprostypes

			IPS = (setdevips & setcooips)
			IPS_union =(setdevips | setcooips)
			#ip_feature, niprows = ip_feature_vector(IPS)
			#if niprows >0:
			#	meaniprows=ip_feature/np.float_(niprows)
			#else:
			#	meaniprows=ip_feature

			Pros_insec= (setdevpros & setcoopros)
			Pros_union = (setdevpros | setcoopros)
			#miips=set() 
			#for ip in IPS:
			#	if len(ValIPDevice.get(ip, set())) <= 10 and len(ValIPCookie.get(ip, set())) <= 25:
			#		miips.add(ip)

			row.append(len(ValDeviceIP.get(device,set())))
			row.append(len(ValDevicePro.get(device, set())))
			row.append(ValDeviceView.get(device))

			row.append(len(ValCookieIP.get(cookie, set())))
			row.append(len(ValCookiePro.get(cookie, set())))
			row.append(ValCookieView.get(cookie))

			row.append(len(IPS))
			row.append(np.sqrt(len(IPS) + alpha) / np.sqrt(len(IPS_union) + alpha))

			row.append(len(ProTypes))
			row.append(np.sqrt(len(ProTypes) + alpha) /  np.sqrt(len(ProTypes_union) + alpha))

			#row.append(len(miips))

			row.append(len(Pros_insec))
			row.append(np.sqrt(len(Pros_insec) + alpha) / np.sqrt(len(Pros_union)+ alpha))

			#row = np.concatenate((row, ip_feature), axis = 0)
			#row = np.concatenate((row, meaniprows), axis = 0)
			
			dataSet.append(row)
	return dataSet, numpatterns

def trainXGBoost(xtr,ytr,rounds,eta,xtst,ytst):
    xgmat = xgb.DMatrix( xtr, label=ytr)
    xgmat2 = xgb.DMatrix( xtst, label=ytst)
    param = {}
    param['eta'] = eta
    param['max_depth'] = 15
    param['subsample'] = 1.0
    param['nthread'] = 12
    param['min_child_weight']=4
    param['gamma']=5.0
    param['colsample_bytree']=1.0
    param['silent']=1
    param['objective'] = 'binary:logistic'
    param['eval_metric']='error'
    watchlist = [ (xgmat,'train') ,(xgmat2,'test')]
    num_round = rounds
    bst = xgb.train( param, xgmat, num_round, watchlist);
    return bst

def predictXGBoost(X,bst):
    xgmat = xgb.DMatrix( X)
    return bst.predict(xgmat)


def training(X_TR, Y_TR):
	NFOLDS=8
	skf = sklearn.cross_validation.KFold(len(X_TR),n_folds=NFOLDS,random_state=0)
	classifiers=list()
	predict_result = []
	for (train,test) in skf:
		iteration=iteration+1
		XvalTR = X_TR[train,:]
		YvalTR = Y_TR[train,]
		XvalTST = X_TR[test,:]
		YvalTST = Y_TR[test,]
		bst = trainXGBoost(XvalTR,YvalTR,200,1.50,XvalTST,YvalTST)
	    classifiers.append((bst,traindev,testdev))
	    y_predict = predictXGBoost(XvalTST, bst)
	    predict_result.append(y_predict)
    return classifiers, predict_result

if __main__():
	df = pd.read_csv('CrossWise_deviceid_result.csv')
	# device_id, id_type, timestamp, user agent, ip_string,country, province, city, WIFI/else, app name, app key, media_id,  time zone
	df.columns = ['device_id','id_type','timestamps','user_agent','ip_string','country','province','city','WIFI/ELSE','app_name','app_key','media_id','time_zone']
	pairs = pd.read_csv('baidu_cookie_mid_for_cross_wise.csv')
	Devices = df.device_id.unique().tolist()
	Cookies = pairs.cookie.unique().tolist()

	ValCookieHandle, ValDevHandle = id_handle('baidu_cookie_mid_for_cross_wise.csv')
	ValHandleCookie, ValHandleDevice = handle_id('baidu_cookie_mid_for_cross_wise.csv')
	df = df.set_index('device_id')
	ValDeviceIP ,ValIPDevice = device_ip_list(Devices, 'CrossWise_deviceid_result.csv')
	ValCookieIP, ValIPCookie = cookie_ip_list(Cookies, 'CrossWise_cookie.csv')
	Cookies= ValCookieIP.keys() #这个是因为跑数导致存在gap
	device_brand, device_family,device_model, device_os_family,device_user_agent_family = parser_device_ua()
	cookie_brand, cookie_family,cookie_model, cookie_os_family,cookie_user_agent_family = parser_cookie_ua()

	ValDeviceView = get_device_view('CrossWise_deviceid_result.csv')
	ValCookieView = get_cookie_view('CrossWise_cookie.csv')

	ValDevicePro = device_pro_list('CrossWise_deviceid_result.csv')
	ValCookiePro = cookie_pro_list('CrossWise_cookie.csv')

	negative_sample_rule = create_negative_sample_rule(Devices, Cookies, ValCookieIP, ValDeviceIP, ValDevHandle, ValCookieHandle, 1.0)
	negative_sample_random_1 = create_negative_sample_random(Devices, Cookies, ValCookieIP, ValDeviceIP, ValDevHandle, ValCookieHandle)

	positive_sample_random = create_positive_sample('./baidu_cookie_mid_for_cross_wise_filter.csv')

	property_type =  property_type_dict()

	dataSet_positive, numpatterns = create_dataset(positive_sample_random, ValDeviceIP, ValDevicePro, ValCookieIP, ValCookiePro, ValIPDevice, ValIPCookie, ValCookieView, ValDeviceView,property_type, 1.0)
	dataSet_negative, numpatterns_negative = create_dataset(dictMerged2, ValDeviceIP, ValDevicePro, ValCookieIP, ValCookiePro, ValIPDevice, ValIPCookie, ValCookieView, ValDeviceView,property_type, 1.0)

	dataTR = np.array(dataSet_positive + dataSet_negative)
	Y_positive = np.ones(dataSet_positive.shape[0])
	Y_negative = np.zeros(dataSet_negative.shape[0])
	YTR = np.array(list(Y_positive) + list(Y_negative))

	df_temp = pd.DataFrame(dataTR)
	for i in range(12):
		df_temp[i] = df_temp[i].fillna(df_temp[i].mean())
	dataTR = df_temp.as_matrix()

	X_TR, X_TST, Y_TR, Y_TST = train_test_split(dataTR,YTR,test_size=0.2)

	classifiers, predict_result = training(X_TR, Y_TR)

