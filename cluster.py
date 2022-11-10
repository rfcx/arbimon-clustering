import os
import json
import shutil
import getopt
import boto3
import time
import numpy as np
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import umap
import pacmap
# import hdbscan
from db import connect
import sqlalchemy as sqal
import datetime as dt
scaler = StandardScaler() # data scaler

session, engine, metadata = connect() # RDS connection
print('DB connections...')
jobs = sqal.Table('jobs', metadata, autoload=True, autoload_with=engine)

log_filename = '_log.json'
progress = 0

def downloadDirectoryFroms3(bucket_name, s3_dir, local_dir, s3_resource):
    bucket = s3_resource.Bucket(bucket_name) 
    for obj in bucket.objects.filter(Prefix = s3_dir):
        if not obj.key.split('.')[-1]=='npy':
            continue
        print('(debug) downloading: ', obj.key)
        bucket.download_file(obj.key, local_dir+obj.key.split('/')[-1])

def cluster(data, eps, min_pts, method='dbscan', metric='euclidean', stdz=True):
    if stdz:
        data = scaler.fit_transform(data)
    if method == 'dbscan':
        clust = DBSCAN(eps = eps, 
                       min_samples = min_pts, 
                       metric = metric).fit(data)
    elif method == 'hdbscan':
        clust = hdbscan.HDBSCAN(min_samples = min_pts,
                                allow_single_cluster = False,
                                metric = metric).fit(data)
    else:
        print('Error: unknown input method.')
    return clust


if __name__ == "__main__":

    t0 = time.time()

    #--- job invoke input
    opts, args = getopt.getopt(os.sys.argv[1:], 'e:m:s:j:a:')
    for opt, arg in opts:
        if opt in ("-e", "--eps"):
            epsilon = float(arg)
        elif opt in ("-m", "--minsamps"):
            min_pts = int(arg)
        elif opt in ("-s", "--maxsize"):
            max_cluster_size = int(arg)
        elif opt in ("-j", "--job_id"):
            job_id = int(arg)
        elif opt in ("-a", "--aed_job_id"):
            aed_job_id = int(arg)

    print('job_id: '+str(job_id))
    print('aed_job_id: '+str(aed_job_id))
    print('eps: '+str(epsilon))
    print('min. pts: '+str(min_pts))

    method = 'dbscan' # clustering algorithm (dbscan or hdbscan)
    
    bucket = 'arbimon2' # where job results will be stored
    print(bucket)
    s3_aed_folder = 'audio_events/'+os.environ.get('DEV_OR_PROD')+'/detection/'+str(aed_job_id)+'/'
    s3_out_folder = 'audio_events/'+os.environ.get('DEV_OR_PROD')+'/clustering/'+str(job_id)+'/'
    localdir = './tmp/'
    if not os.path.exists(localdir):
        os.mkdir(localdir)
    else:
        shutil.rmtree(localdir)
        os.mkdir(localdir)
        
    #--- cloud storage connection
    if os.environ.get('AWS_ACCESS_KEY_ID'):
        s3 = boto3.resource('s3', 
                            aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID'), 
                            aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY'))
    else:
        s3 = boto3.resource('s3')

    print('initialized... ', time.time()-t0)
    progress = 1 # preparation completed
    upd = jobs.update(jobs.c.job_id==job_id).values(progress=jobs.c.progress+1,
                                                    last_update=dt.datetime.now())
    session.execute(upd)
    session.commit()

    
    
    
    t0 = time.time()
    #--- download aed features
    print('Downloading...')
    print('Bucket: ', bucket)
    print('s3_aed_folder: ', s3_aed_folder)
    print('localdir: ', localdir)
    print('access_key_id 0-3: ', os.environ.get('AWS_ACCESS_KEY_ID')[:3])
    print('secret_access_key 0-3: ', os.environ.get('AWS_SECRET_ACCESS_KEY')[:3])
    downloadDirectoryFroms3(bucket, s3_aed_folder, localdir, s3)
    print(len(os.listdir(localdir)), ' feature files downloaded.')
    
    #--- load feature data
    print('Loading features...')
    feas = [] # contains features from aed job
    ids = [] # contains aed_ids from database
    for i in sorted(os.listdir(localdir)):
        try:
            if i.endswith('_features.npy'):
                feas.append(np.load(localdir+'/'+i))
            elif i.endswith('_ids.npy'):
                ids.append(np.load(localdir+'/'+i))
            else:
                continue
        except Exception as e:
            print(i)

    #--- check for no AEDs:
    if len(feas)==0:
        upd = jobs.update(jobs.c.job_id==job_id).values(state='error',last_update=dt.datetime.now())
        session.execute(upd)
        session.commit()
        os.sys.exit('No AEDs in dataset')

    feas = np.vstack(feas)
    ids = np.hstack(ids)
    print(feas.shape)
    print(ids.shape)
    # feas columns:
        # 0 time of day x coord
        # 1 time of day y coord
        # 2 low freq
        # 3 high freq
        # 4 start time
        # 5 end time
        # 6 recording id
        # 7 HOG features
        
    # reduce HOG features to 2D map
    hogmap = pacmap.PaCMAP(n_components=2).fit_transform(scaler.fit_transform(feas[:,5:]))
                                                         
    print('Data loaded...', time.time()-t0)
    progress = 2 # data loaded
    upd = jobs.update(jobs.c.job_id==job_id).values(progress=jobs.c.progress+1,
                                                    last_update=dt.datetime.now())
    session.execute(upd)
    session.commit()




    #--- cluster
    #print('clustering by time of day...')
    #t0 = time.time()
    #clust = MiniBatchKMeans(n_clusters=2, init_size=int(feas.shape[0]*0.05), random_state=1).fit(scaler.fit_transform(feas[:,:2]))
    #print('\t',time.time() - t0)
    
    print('Clustering...')
    t0 = time.time()
    inpt = np.hstack([feas[:,:2], # min and max frequency
                      np.array(feas[:,3]-feas[:,2])[...,np.newaxis], # duration
                      hogmap]) # shape features
    clust = cluster(inpt, 
                    eps=epsilon, 
                    min_pts=min_pts, 
                    method=method)
    clust = clust.labels_
    print('\t',time.time() - t0)
    
    print('Number pts: '+str(len(clust)))
    print('Clustered pts: '+str(len(clust[clust!=-1])))
    print('Number clusters: '+str(len(set(clust).difference([-1]))))
    print('Number noise: '+str(len(clust[clust==-1])))

    feas = feas[clust!=-1,:]
    inpt = inpt[clust!=-1]
    ids = ids[clust!=-1]
    hogmap = hogmap[clust!=-1]
    clust = clust[clust!=-1]

    # limit clusters
    print('Limiting cluster sizes...')
    t0 = time.time()
    tmp1 = []
    tmp2 = []
    tmp3 = []
    tmp4 = []
    tmp5 = []
    for i in list(set(clust)):
        tmp1.append(feas[clust==i][:max_cluster_size])
        tmp2.append(ids[clust==i][:max_cluster_size])
        tmp3.append(hogmap[clust==i][:max_cluster_size])
        tmp4.append(clust[clust==i][:max_cluster_size])
        tmp5.append(inpt[clust==i][:max_cluster_size])
        
    if len(tmp1)==0:
        upd = jobs.update(jobs.c.job_id==job_id).values(state='error',last_update=dt.datetime.now())
        session.execute(upd)
        session.commit()
        os.sys.exit('No clusters found')
        
    feas = np.vstack(tmp1)
    ids = np.hstack(tmp2)
    hogmap = np.vstack(tmp3)
    clust = np.hstack(tmp4)
    inpt = np.vstack(tmp5)
    del tmp1, tmp2, tmp3, tmp4, tmp5
    print('\t',time.time() - t0)
    
    inpt = scaler.fit_transform(inpt)
    
    progress = 3 # clustering completed
    upd = jobs.update(jobs.c.job_id==job_id).values(progress=jobs.c.progress+1,
                                                    last_update=dt.datetime.now())
    session.execute(upd)
    session.commit()
    
    # store aed metadata
    jsn = {
        'aed_id':[int(i) for i in ids],
        'recording_id':[int(i) for i in feas[:,4]],
        'freq_low':[float(i) for i in feas[:,0]],
        'freq_high':[float(i) for i in feas[:,1]],
        'time_min':[float(i) for i in feas[:,2]],
        'time_max':[float(i) for i in feas[:,3]]
    }
    file = str(job_id)+'_aed_info.json'
    with open(file, 'w') as f:
        json.dump(jsn, f)
    s3.Bucket(bucket).upload_file(file, s3_out_folder+file)
    os.remove(file)

    
    
    
    #--- projection
    t0 = time.time()

    print('Projection...')

    # LDA
    mp = LinearDiscriminantAnalysis(n_components=2).fit_transform(inpt, y=clust)
    
    # sort clusters
    print('Sorting clusters by projection...')
    t0 = time.time()
    labels = np.zeros((len(set(clust))))
    centroids = np.zeros((len(set(clust)), mp.shape[1]))
    for c,i in enumerate(list(set(clust))):
        centroids[c,:] = mp[clust==i].mean(axis=0)
        labels[c] = i 
    tmp = umap.UMAP(n_components=1).fit_transform(scaler.fit_transform(centroids)).flatten()
    labeldict = dict(zip(labels[np.argsort(tmp)], range(len(labels))))
    clust = [labeldict[i] for i in clust]
    del tmp, centroids, labels, labeldict
    print('\t',time.time() - t0)
    
    jsn = {
        'aed_id':[int(i) for i in ids],
        'x_coord':[float(i) for i in mp[:,0]],
        'y_coord':[float(i) for i in mp[:,1]],
        'cluster':[int(i) for i in clust],
    }
    file = str(job_id)+'_lda.json'
    with open(file, 'w') as f:
        json.dump(jsn, f)
    s3.Bucket(bucket).upload_file(file, s3_out_folder+file)
    os.remove(file)

    print('Projection finished... ', time.time()-t0)
    progress = 4 # projection completed
    upd = jobs.update(jobs.c.job_id==job_id).values(progress=jobs.c.progress+1,
                                                    last_update=dt.datetime.now(),
                                                    completed=1)
    session.execute(upd)
    session.commit()
    session.close()
    
    shutil.rmtree(localdir)
    print('Done')



