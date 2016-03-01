def generate_negative_sample(device, cookie,id_ip,ValDevHandle,ValCookieHandle):
negative_sample = []
for d in device[23:45]:
	max_jacd = -1
	best_candidate = '0'
	for c in cookie[5000:60000]:
		print d,c
		if ValDevHandle.get(d, dict().keys()) != ValCookieHandle.get(c, dict().keys()):
			ip_union = set(id_ip.get(d).keys())| set(id_ip.get(c).keys())
			m_ip_pv_vec = []
			c_ip_pv_vec = []
			for ip in ip_union:
				m_ip_pv_vec.append(id_ip.get(d).get(ip,[0])[0])
				c_ip_pv_vec.append(id_ip.get(c).get(ip,[0])[0])
			temp_jacd = jaccard_distance(m_ip_pv_vec,c_ip_pv_vec)
			print temp_jacd#,m_ip_pv_vec, c_ip_pv_vec
			if temp_jacd > max_jacd:
				max_jacd = temp_jacd
				best_candidate = c
	negative_sample.append([d,c])
	return negative_sample


def load_ip_info(ipfile):
IPDev=defaultdict(set)
IPCoo = dict()
DeviceIPS = dict()
CookieIPS = dict()
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
		IPDev[k] = id
else:
	for k in ValIPS.keys():
		IPCoo[k] =id
	return id_ip,IPDev, IPCoo


id_ip,IPDev, IPCoo = load_ip_info('id_all_ip.csv')



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






def candidate_generation_1(device,cookie, id_ip,IPDev,IPCoo):
	Candidates = dict()
	for d in device:
		candidatestotal = set()
		device_ips = id_ip.get(d,dict()).keys()
		for ip in device_ips:
			if(XIPS.get(ip)[0] == 0) and len(IPDev.get(ip,set()) + IPCoo.get(ip,set())) <=30:
				candidates = IPCoo.get(ip)
				for candidate in candidates:
					candidatestotal.add(candidate)
		if len(candidatestotal) == 0:
			for ip in device_ips:
				if len(IPDev.get(ip,set()) + IPCoo.get(ip,set())) <=30:
					candidates = IPCoo.get(ip)
					for candidate in candidates:
						candidatestotal.add(candidate)
		if len(candidatestotal) == 0:
			ip_size = dict()
			for ip in device_ips:
				if XIPS.get(ip)[0] ==0:
					ip_size[ip] = len(IPDev.get(ip)) + len(IPCoo.get(ip))
			ip_size = sorted(ip_size.items(), lambda x, y: cmp(x[1], y[1]))
			ips = ip_sizes.keys()[:5]
			for ip in ips:
				candidates = IPCoo.get(ip)
				for candidate in candidates:
					candidatestotal.add(candidate)
		if len(candidatestotal) == 0:
			ip_size = dict()
			for ip in device_ips:
				ip_size[ip] = len(IPDev.get(ip)) + len(IPCoo.get(ip))
			ip_size = sorted(ip_size.items(), lambda x, y: cmp(x[1], y[1]))
			ips = ip_sizes.keys()[:5]
			for ip in ips:
				candidates = IPCoo.get(ip)
				for candidate in candidates:
					candidatestotal.add(candidate)	
	Candidates[device]=candidatestotal
	return Candidates