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
from airflow.operators.python_operator import PythonOperator
from airflow.hooks.base_hook import BaseHook
from airflow.contrib.operators import mlengine_operator
from airflow.contrib.operators import mlengine_operator_utils
from airflow.contrib.hooks.gcp_mlengine_hook import MLEngineHook
from airflow.contrib.hooks.gcs_hook import GoogleCloudStorageHook
from airflow.operators import bash_operator
from airflow.utils import trigger_rule
from airflow import models
import airflow
from airflow import DAG

import datetime
from airflow.models import Variable


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

# GCS bucket names and region, can also be changed.
BUCKET = 'gs://dl-cpu' 
REGION = 'us-east1'

# The code package name comes from the model code in the wals_ml_engine
# directory of the solution code base.
PACKAGE_URI = BUCKET + '/code/cpu_train_5days_train_TF11.zip'
# JOB_DIR = BUCKET + '/jobs'

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

dag = models.DAG('ml_test_training_6hr_get_fileName_v1', default_args=default_args,
          schedule_interval=schedule_interval)

dag.doc_md = __doc__


#
#
# Task Definition
#
#
lobs = ["six_hour","12","24","48","96","128"] 
Myparams={'job_dir':'gs://dl-cpu/jobs/recserve_lstm_test_2/',
         'URL': 'gs://dl-cpu/jobs/recserve_lstm_test_1/model*',
         'TRAIN_FILES':BUCKET+'/cpu_training_data/cpu_'+lobs[0]+'_test_lstm_small.tfrecords', 
         'TEST_FILES':BUCKET+'/cpu_training_data/cpu_'+lobs[0]+'_test_lstm_small.tfrecords',
         }
job_id = Myparams["job_dir"]+'_{0}'.format(datetime.datetime.now().strftime('%Y%m%d%H%M'))
output_dir = BUCKET+'/airflowoutput/'

#
####get the latest wrote weights file
#
templated_command = """
filename=$(gsutil ls -l "{{ params.URL }}" | sort -k2n | tail -n1 | awk 'END {$1=$2=""; sub(/^[ \t]+/, ""); print }')
fname="${filename%.*}" 
echo $fname
"""

t0 = bash_operator.BashOperator(
    task_id='get_latest_weights',
    depends_on_past=False,
    bash_command=templated_command,
    xcom_push=True,
    params={'URL': Myparams['URL']},
    dag=dag
)


#######Tried pull Xcom from pythonOperator and bashOperator##############
# def pull_function(**context):
#      weights = context['task_instance'].xcom_pull(task_ids='get_latest_weights')
#      return weights



# pull = airflow.operators.PythonOperator(
#     task_id='pull', dag=dag, python_callable=pull_function,provide_context=True)

# # pull = BashOperator(
# #     task_id='pull',
# #     bash_command="echo {{ task_instance.xcom_pull(task_ids='get_latest_weights') }}",
# #     xcom_push=True,
# #     dag=dag)
# t0.set_downstream(pull)


##############################################################


# ML Engine training job#

training_args_0 = [
                 '--job-dir', Myparams["job_dir"],
                 '--train-files', Myparams['TRAIN_FILES'],
                 '--eval-files', Myparams['TEST_FILES'],
                 '--test-files', Myparams['TEST_FILES'],
                 '--train-step', '1000',
                 '--num-epochs', '100',
                 '--train-batch-size','7830',
                 '--train-size','7830',
                 '--hidden-units', "460,220,40",
                 '--train-input-size','72',
                 '--eval-every-secs','300',
                  '--fix-flag','1',
                 '--checkpoint-path',"{{task_instance.xcom_pull(task_ids='get_latest_weights')}}"
                 #    "(task_ids='get_latest_weights') }}"
                 # '--checkpoint-path','{0}'.format("{{ task_instance.xcom_pull"
                 #    "(task_ids='get_latest_weights') }}")
                ]

t1 = mlengine_operator.MLEngineTrainingOperator(   
    # gcp_conn_id='project_connection',
    task_id='ml_engine_training_op',
    project_id=PROJECT_ID,
    job_id=job_id,
    package_uris=[PACKAGE_URI],
    training_python_module='trainer.task',
    training_args=training_args_0,
    region=REGION,
    runtime_version='1.9',
    scale_tier="BASIC",
    # master_type='complex_model_m',
    dag=dag
)
t0.set_downstream(t1)


# job_id_1 = 'recserve_lstm_test_1{0}'.format(datetime.datetime.now().strftime('%Y%m%d%H%M'))
# training_args_1 = ['--job-dir', job_dir,
#                  '--train-files', TRAIN_FILES,
#                  '--eval-files', EVAL_FILES,
#                  '--test-files', TEST_FILES,
#                  '--train-step', '1000',
#                  '--num-epochs', '20',
#                  '--train-batch-size','7830',
#                  '--train-size','7830',
#                  '--hidden-units', "450,200,30",
#                 '--train-input-size','72',
#                 '--fix-flag','0',
#                 '--eval-every-secs','300',
#                 '--checkpoint-path','gs://dl-cpu/cpu_train_6hrs_diurnal_1300Machine_DeepNetwork_xavierInitializer/model.ckpt-45517'
#                  ]
# t2 = MLEngineTrainingOperator(
#     # gcp_conn_id='project_connection',
#     task_id='ml_engine_training_op_1',
#     project_id=PROJECT_ID,
#     job_id=job_id_1,
#     package_uris=[PACKAGE_URI],
#     training_python_module='trainer.task',
#     training_args=training_args_1,
#     region=REGION,
#     scale_tier='CUSTOM',
#     master_type=,complex_model_m',
#     dag=dag
# )





