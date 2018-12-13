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
from airflow import DAG
from airflow.hooks.base_hook import BaseHook
from airflow.operators.ml_engine_plugin import MLEngineTrainingOperator

import datetime

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
# PROJECT_ID="metal-being-221615"
# Data set constants, used in BigQuery tasks.  You can change these
# to conform to your data.

# GCS bucket names and region, can also be changed.
BUCKET = 'gs://dl-cpu' 
REGION = 'us-east1'

# The code package name comes from the model code in the wals_ml_engine
# directory of the solution code base.
PACKAGE_URI = BUCKET + '/code/cpu_train_5days_train_TF11.zip'
JOB_DIR = BUCKET + '/jobs'

default_args = {
    'owner': 'lori',
    'depends_on_past': False,
    'start_date': airflow.utils.dates.days_ago(2),
    'email': ['lori.wan24@gmail.com'],
    'email_on_failure': True,
    'email_on_retry': True,
    # 'retries': 1,
    # 'retry_delay': datetime.timedelta(minutes=5)
}

# Default schedule interval using cronjob syntax - can be customized here
# or in the Airflow console.
schedule_interval = '@once'

dag = DAG('ml_test_training_6hr_dag', default_args=default_args,
          schedule_interval=schedule_interval)

dag.doc_md = __doc__


#
#
# Task Definition
#
#

job_id = 'recserve_lstm_test_1{0}'.format(datetime.datetime.now().strftime('%Y%m%d%H%M'))
job_dir = BUCKET + '/jobs/' + 'recserve_lstm_test_1/'
output_dir = BUCKET+'/airflowoutput/'


# PACKAGE_PATH=BUCKET+'/code/cpu_train_5days_train_TF11.zip' # this can be a gcs location to a zipped and uploaded package
TRAIN_FILES=BUCKET+'/cpu_training_data/cpu_sixhour_test_lstm_small.tfrecords'
EVAL_FILES=BUCKET+'/cpu_training_data/cpu_sixhour_test_lstm_small.tfrecords'
TEST_FILES=BUCKET+'/cpu_training_data/cpu_sixhour_test_lstm_small.tfrecords'

training_args_0 = ['--job-dir', job_dir,
                 '--train-files', TRAIN_FILES,
                 '--eval-files', EVAL_FILES,
                 '--test-files', TEST_FILES,
                 '--train-step', '1000',
                 '--num-epochs', '100',
                 '--train-batch-size','7830',
                 '--train-size','7830',
                 '--hidden-units', "450,200,30",
                '--train-input-size','72',
                '--fix-flag','1',
                '--eval-every-secs','300',
                '--checkpoint-path','gs://dl-cpu/cpu_train_6hrs_diurnal_1300Machine_DeepNetwork_xavierInitializer/model.ckpt-45517'
                 ]

t1 = MLEngineTrainingOperator(   
    # gcp_conn_id='project_connection',
    task_id='ml_engine_training_op',
    project_id=PROJECT_ID,
    job_id=job_id,
    package_uris=[PACKAGE_URI],
    training_python_module='trainer.task',
    training_args=training_args_0,
    region=REGION,
    scale_tier='CUSTOM',
    master_type='complex_model_m',
    dag=dag
)

# ML Engine training job

job_id_1 = 'recserve_lstm_test_1{0}'.format(datetime.datetime.now().strftime('%Y%m%d%H%M'))
training_args_1 = ['--job-dir', job_dir,
                 '--train-files', TRAIN_FILES,
                 '--eval-files', EVAL_FILES,
                 '--test-files', TEST_FILES,
                 '--train-step', '1000',
                 '--num-epochs', '20',
                 '--train-batch-size','7830',
                 '--train-size','7830',
                 '--hidden-units', "450,200,30",
                '--train-input-size','72',
                '--fix-flag','0',
                '--eval-every-secs','300',
                '--checkpoint-path','gs://dl-cpu/cpu_train_6hrs_diurnal_1300Machine_DeepNetwork_xavierInitializer/model.ckpt-45517'
                 ]
t2 = MLEngineTrainingOperator(
    # gcp_conn_id='project_connection',
    task_id='ml_engine_training_op_1',
    project_id=PROJECT_ID,
    job_id=job_id_1,
    package_uris=[PACKAGE_URI],
    training_python_module='trainer.task',
    training_args=training_args_1,
    region=REGION,
    scale_tier='CUSTOM',
    master_type='complex_model_m',
    dag=dag
)


t1.set_downstream(t2)
# t2.set_upstream(t1)

