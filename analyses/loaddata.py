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

# Set the filepath
filepath = '..'

# Specify the collection name in Firestore
collname = 'emotionconexp'

# Initialize the Firebase Admin SDK with the provided credentials
try: 
    cred = credentials.Certificate('emotioncon2-firebase-adminsdk-kmpni-88fe22d6a2.json')
    default_app = firebase_admin.initialize_app(cred)
except: 
    print('Seems certificate already loaded')

# Create a Firestore client
client = firestore.client()

# Iterate over the documents in the collection
for subj in client.collection(collname).stream():
    # Get the document ID and convert it to a dictionary
    iddoc = client.document(collname + '/{0}'.format(subj.id)).get().to_dict()
    subjid = str(iddoc['subjectID_prolific'])
    fsid = str(iddoc['uid_firebase'])
    
    try: 
        # Skip documents that are not marked as completed
        if not iddoc['completed']:
            continue
    except: 
        continue
    
    try: 
        # Convert the start_time_jspsych field to a timestamp
        iddoc['start_time_jspsych'] = iddoc['start_time_jspsych'].timestamp()
    except: 
        pass
    
    # Get the setup variables and task data for the document
    setupdoc = client.document(collname + '/{0}/setup/variables'.format(subj.id)).get().to_dict()
    datadoc = client.document(collname + '/{0}/taskdata/data'.format(subj.id)).get().to_dict()
    
    # Merge all the data into a single dictionary
    alldoc = {**iddoc, **setupdoc, **datadoc}
    print(subjid + ' downloaded')
    
    # Write the merged data to a file
    f = open(filepath + '/data/' + collname + '_' + subjid + '_' + fsid + '_main.txt', 'w')
    f.write(json.dumps(alldoc))
    f.close()

    try: 
        # Get the datadump field and save it as a file
        datadump = client.document(collname + '/{0}/taskdata/datadump'.format(subj.id)).get().to_dict()
        f = open(filepath + '/data/' + collname + '_' + subjid + '_' + fsid + '_datadump.txt', 'w')
        td = pd.DataFrame(json.loads(datadump['jsondatadump']))
        f.write(json.dumps(td.to_json()))
        f.close()
        print(subjid + ' / ' + fsid + ' datadump saved.')
    except: 
        print(subjid + ' / ' + fsid + ' *** datadump failed ***')

    try: 
        # Get the timeline field and save it as a file
        timeline = client.document(collname + '/{0}/setup/timeline'.format(subj.id)).get().to_dict()
        f = open(filepath + '/data/' + collname + '_' + subjid + '_' + fsid + '_timeline.txt', 'w')
        td = pd.DataFrame(json.loads(timeline['timeline']))
        f.write(json.dumps(td.to_json()))
        f.close()
        print(subjid + ' / ' + fsid + ' timeline saved.')
    except: 
        print(subjid + ' / ' + fsid + ' *** timeline failed ***')

    try: 
        # Get the questionnaire data and save it as a file
        quests = client.document(collname + '/{0}/questionnaire/data'.format(subj.id)).get().to_dict()
        f = open(filepath + '/data/' + collname + '_' + subjid + '_' + fsid + '_questionnairedata.txt', 'w')
        f.write(json.dumps(quests))
        f.close()
        print(subjid + ' / ' + fsid + ' questionnaire data saved.')
    except: 
        print(subjid + ' / ' + fsid + ' *** questionnaire data failed ***')
