import numpy as np
import numpy.matlib as nm
from scipy.linalg import expm
from matplotlib.pyplot import *
import firebase_admin
from firebase_admin import credentials, firestore, db
import json
from IPython.core.debugger import set_trace
from scipy import stats
np.set_printoptions(threshold=np.inf)
import pandas as pd
pd.set_option("display.max_rows", None, "display.max_columns", None)
ion()

filepath = '..'
collname = 'emotionconexp'

try: 
    cred = credentials.Certificate('emotioncon2-firebase-adminsdk-kmpni-88fe22d6a2.json')
    default_app = firebase_admin.initialize_app(cred)
except: 
    print('Seems certificate already loaded')

client = firestore.client()
for subj in client.collection(collname).stream():
    iddoc = client.document(collname + '/{0}'.format(subj.id)).get().to_dict()
    subjid = str(iddoc['subjectID_prolific'])
    fsid = str(iddoc['uid_firebase'])
    try: 
        if not iddoc['completed']:
            continue
    except: 
        continue
    try: 
        iddoc['start_time_jspsych'] = iddoc['start_time_jspsych'].timestamp()
    except: 
        pass 
    setupdoc = client.document(collname + '/{0}/setup/variables'.format(subj.id)).get().to_dict()
    datadoc = client.document(collname + '/{0}/taskdata/data'.format(subj.id)).get().to_dict()
    alldoc = {**iddoc,**setupdoc,**datadoc}
    print(subjid+' downloaded')
    f=open(filepath + '/data/'+collname+'_'+subjid+'_'+fsid+'_main.txt','w')
    f.write(json.dumps(alldoc))
    f.close()

    try: 
        datadump = client.document(collname + '/{0}/taskdata/datadump'.format(subj.id)).get().to_dict()
        f=open(filepath + '/data/'+collname+'_'+subjid+'_'+fsid+'_datadump.txt','w')
        #f.write(json.dumps(datadump['jsondatadump']))
        td = pd.DataFrame(json.loads(datadump['jsondatadump']))
        f.write(json.dumps(td.to_json()))
        f.close()
        print(subjid+' / '+fsid+' datadump saved.')
    except: 
        print(subjid+' / '+fsid+'*** datadump failed ***')

    try: 
        timeline = client.document(collname + '/{0}/setup/timeline'.format(subj.id)).get().to_dict()
        f=open(filepath + '/data/'+collname+'_'+subjid+'_'+fsid+'_timeline.txt','w')
        #f.write(json.dumps(datadump['jsondatadump']))
        td = pd.DataFrame(json.loads(timeline['timeline']))
        f.write(json.dumps(td.to_json()))
        f.close()
        print(subjid+' / '+fsid+' timeline saved.')
    except: 
        print(subjid+' / '+fsid+'*** timeline failed ***')

    try: 
        quests = client.document(collname + '/{0}/questionnaire/data'.format(subj.id)).get().to_dict()
        f=open(filepath + '/data/'+collname+'_'+subjid+'_'+fsid+'_questionnairedata.txt','w')
        #td = pd.DataFrame(json.loads(quests))
        f.write(json.dumps(quests))
        f.close()
        print(subjid+' / '+fsid+' questionnaire data saved.')
    except: 
        print(subjid+' / '+fsid+'*** questionnaire data failed ***')
