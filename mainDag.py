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
from subDag import sub_dag
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
parent_dag_name = 'DummyMain'

child_dag_name='echoSub'

dag = models.DAG(parent_dag_name, default_args=default_args,
          schedule_interval=schedule_interval)

dag.doc_md = __doc__

t0 = bash_operator.BashOperator(
         task_id='start',
         depends_on_past=False,
         bash_command="echo hello world",
         xcom_push=True,
         dag=dag
     )
subdag = SubDagOperator(
    subdag=sub_dag(parent_dag_name, child_dag_name, default_args, dag.schedule_interval),
    task_id=child_dag_name,
    default_args=default_args,
    dag=dag)


t0.set_downstream(subdag)