import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import pandas as pd ,numpy as np,scipy as sp 
from scipy import spatial
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

###########################################################################################################
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
					# print temp_jacd
					if temp_jacd > max_jacd:
						max_jacd = temp_jacd
						best_candidate = c
		negative_sample.append([d,c])
	return negative_sample

###########################################################################################################
## create the positive candidate by selecting the same handle cookie and device ,and pair them ############
###########################################################################################################
def generate_candidate(cookie_mat,device_train):
	cookie_train = pd.DataFrame(cookie_mat[1:]).ix[:,[0,1]]
	cookie_train.columns = cookie_mat[0][:2]
	temp = device_train.set_index('drawbridge_handle').join(cookie_train.set_index('drawbridge_handle'))
	return temp


def jaccard_distance(c_media_pv, m_mobile_pv):
	return scipy.spatial.distance.jaccard(c_media_pv,m_mobile_pv)


def load_ip_info(ipfile):
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
	return id_ip
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




ValDevHandle = device_id_handle('dev_train_basic.csv')
ValCookieHandle = device_id_handle('cookie_all_basic.csv')
id_ip = load_ip_info('id_all_ip.csv')




