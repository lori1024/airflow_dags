# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DAG definition for recserv model training."""
import airflow
from airflow.operators.python_operator import PythonOperator
from airflow.hooks.base_hook import BaseHook
from airflow.contrib.operators import mlengine_operator
from airflow.contrib.operators import mlengine_operator_utils
from airflow.contrib.hooks.gcp_mlengine_hook import MLEngineHook
from airflow.contrib.hooks.gcs_hook import GoogleCloudStorageHook
from airflow.operators import bash_operator
from airflow.utils import trigger_rule
from airflow import models

from airflow import DAG

import datetime
from airflow.models import Variable

from airflow.operators.subdag_operator import SubDagOperator
from sequential_training_sub import sub_dag
# --------------------------------------------------------------------------------
# Dag Configs
# --------------------------------------------------------------------------------
def _get_project_id():
  """Get project ID from default GCP connection."""

  extras = BaseHook.get_connection('google_cloud_default').extra_dejson
  key = 'extra__google_cloud_platform__project'
  if key in extras:
    project_id = extras[key]
  else:
    raise ('Must configure project_id in google_cloud_default '
           'connection from Airflow Console')
  return project_id

PROJECT_ID = _get_project_id()

default_args = {
    'owner': 'lori',
    'depends_on_past': False,
    'start_date': airflow.utils.dates.days_ago(2),
    'email': ['lori.wan24@gmail.com'],
    'email_on_failure': True,
    'email_on_retry': True,
     'provide_context': True
    # 'retries': 1,
    # 'retry_delay': datetime.timedelta(minutes=5)
}

# Default schedule interval using cronjob syntax - can be customized here
# or in the Airflow console.
schedule_interval = '@once'
# dagTaskName = 'Sequential_diurnal_doubleTime_test'
parent_dag_name = 'initial_training'


dag = models.DAG(parent_dag_name, default_args=default_args,
          schedule_interval=schedule_interval)
dag.doc_md = __doc__

# --------------------------------------------------------------------------------
# Project Parameters
# --------------------------------------------------------------------------------
# GCS bucket names and region, can also be changed.
BUCKET = 'gs://dl-cpu' 
REGION = 'us-east1'

# The code package name comes from the model code in the wals_ml_engine
# directory of the solution code base.
PACKAGE_URI = BUCKET + '/code/cpu_train_5days_train_TF11.zip'
ProjectName = 'Diurnal_doubleTime_sequential_train'
TrainingParams = {
                  'train_id':[6,12,24,48,96,192],
                  'inputSize':[860084,855651,847220,829461,794836,721141],
                  'batchSize':[200000,150000,100000,80000,40000,20000],
                  'hiddenUnits':["200,40,4","300,50,6","500,60,8","800,80,12","1800,150,20","3000,250,30"]
                  }
output_dir = BUCKET+'/airflowoutput/'
ScaleTier = 'BASIC'
NumEpoch=['10','10','20']
ConfigFile=BUCKET +'/bashCode/config_fivedays.yaml'
# --------------------------------------------------------------------------------
# Initial Training
# --------------------------------------------------------------------------------




job_dir0='{}/jobs/{}/Diurnal_sequential_{}hr'.format(BUCKET,ProjectName,str(TrainingParams['train_id'][0]))
TRAIN_FILES0='{}/cpu_training_data/Sequential_diurnal_doubleTime/Diurnal_double_sequential/{}_diurnal_train.tfrecords'.format(BUCKET,str(TrainingParams['train_id'][0]))
TEST_FILE0='{}/cpu_training_data/Sequential_diurnal_doubleTime/Diurnal_double_sequential/{}_diurnal_test.tfrecords'.format(BUCKET,str(TrainingParams['train_id'][0]))
job_id0 = job_dir0+'_initial_{0}'.format(datetime.datetime.now().strftime('%Y%m%d%H%M'))
training_args_0 = [
                 '--job-dir', job_dir0,
                 '--train-files', TRAIN_FILES0,
                 '--eval-files', TEST_FILE0,
                 '--test-files', TEST_FILE0,
                 '--train-step', '1000',
                 '--num-epochs', NumEpoch[0],
                 # '--config',ConfigFile,
                 '--train-batch-size',str(TrainingParams['batchSize'][0]),
                 '--train-size',str(TrainingParams['inputSize'][0]),
                 '--hidden-units', TrainingParams['hiddenUnits'][0],
                 '--train-input-size',str(TrainingParams['train_id'][0]*12),
                 '--eval-every-secs','300',
                  '--fix-flag','0',
                ]

t0 = mlengine_operator.MLEngineTrainingOperator(   
    # gcp_conn_id='project_connection',
    task_id='sequential_startPoint',
    project_id=PROJECT_ID,
    job_id=job_id0,
    package_uris=[PACKAGE_URI],
    training_python_module='trainer.task',
    training_args=training_args_0,
    region=REGION,
    runtime_version='1.9',
    # scale_tier=ScaleTier,
    dag=dag
)
child_dag_names = ['sequential_training_12','sequential_training_24']
subdag = SubDagOperator(
    subdag=sub_dag(parent_dag_name, child_dag_names[0], default_args, dag.schedule_interval),
    task_id=child_dag_names[0],
    default_args=default_args,
    dag=dag)
subdag1 = SubDagOperator(
    subdag=sub_dag(parent_dag_name, child_dag_names[1], default_args, dag.schedule_interval),
    task_id=child_dag_names[1],
    default_args=default_args,
    dag=dag)

t0.set_downstream(subdag)
subdag.set_downstream(subdag1)
