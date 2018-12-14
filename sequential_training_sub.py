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



def sub_dag(parent_dag_name, child_dag_name, args, schedule_interval):
     dag = DAG(
         '%s.%s' % (parent_dag_name, child_dag_name),
         default_args=args,
         start_date=args['start_date'],
         max_active_runs=1,
       )

     # --------------------------------------------------------------------------------
     # Project Parameters
     # --------------------------------------------------------------------------------
     # GCS bucket names and region, can also be changed.
     extras = BaseHook.get_connection('google_cloud_default').extra_dejson
     key = 'extra__google_cloud_platform__project'
     PROJECT_ID = extras[key]
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
     taskIdx = [int(s) for s in child_dag_name.split('_') if s.isdigit()]
     task_idx = TrainingParams['train_id'].index(taskIdx[0])

     # --------------------------------------------------------------------------------
     # Get The trainign Weights file from last training
     # --------------------------------------------------------------------------------
     Pathparams={
              'job_dir':'{}/jobs/{}/Diurnal_sequential_{}hr'.format(BUCKET,ProjectName,str(TrainingParams['train_id'][task_idx])),
              'TRAIN_FILES':'{}/cpu_training_data/Sequential_diurnal_doubleTime/Diurnal_double_sequential/{}_diurnal_train.tfrecords'.format(BUCKET,str(TrainingParams['train_id'][task_idx])), 
              'TEST_FILES':'{}/cpu_training_data/Sequential_diurnal_doubleTime/Diurnal_double_sequential/{}_diurnal_test.tfrecords'.format(BUCKET,str(TrainingParams['train_id'][task_idx])) 
              }
     weight_path='{}/jobs/{}/Diurnal_sequential_{}hr'.format(BUCKET,ProjectName,str(TrainingParams['train_id'][task_idx-1]))

     templated_command = """
     filename=$(gsutil ls -l "{{ params.URL }}" | sort -k2n | tail -n1 | awk 'END {$1=$2=""; sub(/^[ \t]+/, ""); print }')
     fname="${filename%.*}" 
     echo $fname
     """

     t1 = bash_operator.BashOperator(
         task_id='get_latest_weights_'+str(TrainingParams['train_id'][task_idx]),
         depends_on_past=False,
         bash_command=templated_command,
         xcom_push=True,
         params={'URL': weight_path+'/model*'},
         dag=dag
     )
     dagid = '%s.%s' % (parent_dag_name, child_dag_name)
     task_id_t1='get_latest_weights_'+str(TrainingParams['train_id'][task_idx])
     checkPointPathxcom = "{{task_instance.xcom_pull(dag_id = '"+dagid +"',task_ids='"+task_id_t1+"')}}"
     
     
     
     #-----------------------------------------------------------------------------
     # reuse weights training
     #-----------------------------------------------------------------------------
        

     # ML Engine training job#
     job_id = Pathparams["job_dir"]+'fixed_{0}'.format(datetime.datetime.now().strftime('%Y%m%d%H%M'))
     training_args_0 = [
                      # '--config',ConfigFile,
                      '--job-dir', Pathparams["job_dir"],
                      '--train-files', Pathparams['TRAIN_FILES'],
                      '--eval-files', Pathparams['TEST_FILES'],
                      '--test-files', Pathparams['TEST_FILES'],
                      '--train-step', '1000',
                      '--num-epochs', NumEpoch[1],
                      '--train-batch-size',str(TrainingParams['batchSize'][task_idx]),
                      '--train-size',str(TrainingParams['inputSize'][task_idx]),
                      '--hidden-units', TrainingParams['hiddenUnits'][task_idx],
                      '--train-input-size',str(TrainingParams['train_id'][task_idx]*12),
                      '--eval-every-secs','300',
                       '--fix-flag','1',
                      '--checkpoint-path',checkPointPathxcom
                     ]

     t2 = mlengine_operator.MLEngineTrainingOperator(   
         # gcp_conn_id='project_connection',
         task_id='training_with_fixed_weights_'+str(TrainingParams['train_id'][task_idx]),
         project_id=PROJECT_ID,
         job_id=job_id,
         package_uris=[PACKAGE_URI],
         training_python_module='trainer.task',
         training_args=training_args_0,
         region=REGION,
         runtime_version='1.9',
         scale_tier=ScaleTier,
         # master_type='complex_model_m',
         dag=dag
     )
      
     # t1.set_downstream(t2)
     #-----------------------------------------------------------------------------
     # fixed weights training
     #-----------------------------------------------------------------------------
     # ML Engine training job#
     job_id = Pathparams["job_dir"]+'unfixed_{0}'.format(datetime.datetime.now().strftime('%Y%m%d%H%M'))
     training_args_0 = [
                      # '--config',ConfigFile,
                      '--job-dir', Pathparams["job_dir"],
                      '--train-files', Pathparams['TRAIN_FILES'],
                      '--eval-files', Pathparams['TEST_FILES'],
                      '--test-files', Pathparams['TEST_FILES'],
                      '--train-step', '1000',
                      '--num-epochs', NumEpoch[2],
                      '--train-batch-size',str(TrainingParams['batchSize'][task_idx]),
                      '--train-size',str(TrainingParams['inputSize'][task_idx]),
                      '--hidden-units', TrainingParams['hiddenUnits'][task_idx],
                      '--train-input-size',str(TrainingParams['train_id'][task_idx]*12),
                      '--eval-every-secs','300',
                      '--fix-flag','0',
                     ]
     t3 = mlengine_operator.MLEngineTrainingOperator(   
         # gcp_conn_id='project_connection',
         task_id='full_training_'+str(TrainingParams['train_id'][task_idx]),
         project_id=PROJECT_ID,
         job_id=job_id,
         package_uris=[PACKAGE_URI],
         training_python_module='trainer.task',
         training_args=training_args_0,
         region=REGION,
         runtime_version='1.9',
         scale_tier=ScaleTier,
         dag=dag
     )
     t2.set_upstream(t1)
     t3.set_upstream(t2)
     return dag     


