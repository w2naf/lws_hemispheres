#!/usr/bin/env python
"""
mongoexport_to_json.py

This script will export an entire MongoDB database, including all of its collections,
to JSON files. This is needed because the standard mongoexport command will only 
export one collection at a time.
"""

import os
import shutil
import subprocess
import tqdm
import pymongo

db_name     = 'mstid_2016'
mongo_port  = 27017
export_dir  = os.path.join('output','mongoexport_json',db_name)

# Prepare output dictionary.
if os.path.exists(export_dir):
    shutil.rmtree(export_dir)

os.makedirs(export_dir)

mongo   = pymongo.MongoClient(port=mongo_port)
db      = mongo[db_name]
colls   = db.list_collection_names()
mongo.close()

for collection in tqdm.tqdm(colls,dynamic_ncols=True):
    out_path    = os.path.join(export_dir,collection+'.json')
    cmd = 'mongoexport --db={!s} --collection={!s} --out={!s} --pretty'.format(db_name,collection,out_path)
    tqdm.tqdm.write(cmd)
    subprocess.run(cmd,shell=True)
