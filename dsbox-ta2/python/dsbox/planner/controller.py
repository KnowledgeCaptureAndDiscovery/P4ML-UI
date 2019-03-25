import os
import sys
import os.path
import uuid
import copy
import math
import json
import numpy
import shutil
import traceback
import pandas as pd

from dsbox.planner.leveltwo.l1proxy import LevelOnePlannerProxy
from dsbox.planner.leveltwo.planner import LevelTwoPlanner
from dsbox.schema.data_profile import DataProfile
from dsbox.schema.problem_schema import TaskType, TaskSubType, Metric
from dsbox.executer.executionhelper import ExecutionHelper
from dsbox.planner.common.data_manager import Dataset, DataManager
from dsbox.planner.common.pipeline import Pipeline, PipelineExecutionResult
from dsbox.planner.common.problem_manager import Problem
from dsbox.planner.common.resource_manager import ResourceManager
from dsbox.planner.common.primitive import Primitive
from dsbox.planner.levelone.primitives import Primitives, Category, DSBoxPrimitives, D3mPrimitives,D3mPrimitive
from dsbox.planner.levelone.planner import (LevelOnePlanner, get_d3m_primitives, AffinityPolicy)

class Feature:
    def __init__(self, resource_id, feature_name):
        self.resource_id = resource_id
        self.feature_name = feature_name

class Controller(object):
    problem = None
    dataset = None
    execution_helper = None
    resource_manager = None

    config = None
    num_cpus = 0
    ram = 0
    timeout = 60

    exec_pipelines = []
    l1_planner = None
    l2_planner = None

    #create a flag that shows whether there's a restriction file
    # flag = False
    model_name_flag = False
    feature_extraction_flag = False
    imputation_flag = False
    replaced_model_flag = False



    """
    This is the overall "planning" coordinator. It is passed in the data directory
    and the primitives library directory, and it generates plans by calling out to L1, L2
    and L3 planners.
    """
    def __init__(self, libdir):
        # FIXME: This should change to the primitive discovery interface
        self.libdir = os.path.abspath(libdir)

    '''
    Set config directories and data schema file
    '''
    def initialize_from_config(self, config):
        self.config = config

        self.log_dir = self._dir(config, 'pipeline_logs_root', True)
        self.exec_dir = self._dir(config, 'executables_root', True)
        self.tmp_dir = self._dir(config, 'temp_storage_root', True)

        self.num_cpus = int(config.get('cpus', 0))
        self.ram = config.get('ram', 0)
        self.timeout = (config.get('timeout', 60))*60

        # Create some debugging files
        self.logfile = open("%s%slog.txt" % (self.tmp_dir, os.sep), 'w')
        self.errorfile = open("%s%sstderr.txt" % (self.tmp_dir, os.sep), 'w')
        self.pipelinesfile = open("%s%spipelines.txt" % (self.tmp_dir, os.sep), 'w')

        self.problem = Problem()
        self.data_manager = DataManager()
        self.execution_helper = ExecutionHelper(self.problem, self.data_manager)
        self.resource_manager = ResourceManager(self.execution_helper, self.num_cpus)

        # Redirect stderr to error file
        sys.stderr = self.errorfile


    def initialize_from_restriction(self, restriction):
        
        self.restriction = restriction
        self.model_name_flag = False
        self.exclude_model_flag = False
        self.feature_extraction_flag = False
        self.imputation_flag = False
        self.replace_model_flag = False



        if restriction.get('include_model') :
            self.model_name = restriction.get('include_model')
            self.model_name_flag = True

        if restriction.get('include_feature_extraction'):
            self.feature_extraction = restriction.get('include_feature_extraction')
            self.feature_extraction_flag = True

        if restriction.get('use_imputation_method') :
            self.imputation = restriction.get('use_imputation_method')
            self.imputation_flag = True


        if restriction.get('replace_model') :
            self.replace_model = restriction.get('replace_model')
            self.replace_model_flag = True

        if restriction.get('exclude_model') :
            self.exclude_model = restriction.get('exclude_model')
            self.exclude_model_flag = True

        # Redirect stderr to error file
        sys.stderr = self.errorfile

        

    '''
    Set config directories and schema from just problemdir, datadir and outputdir
    '''
    def initialize_simple(self, problemdir, datadir, outputdir):
        self.initialize_from_config({
            "problem_root": problemdir,
            "problem_schema": problemdir + os.sep + 'problemDoc.json',
            "training_data_root": datadir,
            "dataset_schema": datadir + os.sep + 'datasetDoc.json',
            'pipeline_logs_root': outputdir + os.sep + "logs",
            'executables_root': outputdir + os.sep + "executables",
            'temp_storage_root': outputdir + os.sep + "temp",
            "timeout": 60,
            "cpus"  : "4",
            "ram"   : "4Gi"
        })

        # self.restriction(self, restriction)


    """
    Set the task type, metric and output type via the schema
    """
    def load_problem(self):
        problemroot = self._dir(self.config, 'problem_root')
        problemdoc = self.config.get('problem_schema', None)
        assert(problemroot is not None)
        self.problem.load_problem(problemroot, problemdoc)

    """
    Initialize data from the config
    """
    def initialize_training_data_from_config(self):
        dataroot = self._dir(self.config, 'training_data_root')
        datadoc = self.config.get('dataset_schema', None)
        assert(dataroot is not None)
        dataset = Dataset()
        dataset.load_dataset(dataroot, datadoc)
        self.data_manager.initialize_data(self.problem, [dataset], view='TRAIN')

    """
    Initialize all (config, problem, data) from features
    - Used by TA3
    """
    def initialize_from_features(self, datafile, train_features, target_features, outputdir, view=None):
        data_directory = os.path.dirname(datafile)
        self.initialize_simple(outputdir, data_directory, outputdir)

        # Load datasets first
        filters = {}
        targets = {}
        dataset = Dataset()
        dataset.load_dataset(data_directory, datafile)

        if train_features is not None:
            filters[dataset.dsID] = list(map(
                lambda x: {"resID": x.resource_id, "colName": x.feature_name}, train_features
            ))
            self.problem.dataset_filters = filters

        if target_features is not None:
            targets[dataset.dsID] = list(map(
                lambda x: {"resID": x.resource_id, "colName": x.feature_name}, target_features
            ))
            self.problem.dataset_targets = targets

        self.data_manager.initialize_data(self.problem, [dataset], view)


    """
    Initialize the L1 and L2 planners
    """
    def initialize_planners(self):
        self.l1_planner = LevelOnePlannerProxy(self.libdir, self.execution_helper)
        self.l2_planner = LevelTwoPlanner(self.libdir, self.execution_helper)

    """
    Train and select pipelines
    """
    def train(self, planner_event_handler, cutoff=10):

        self.exec_pipelines = []
        self.l2_planner.primitive_cache = {}
        self.l2_planner.execution_cache = {}

        self.logfile.write("Task type: %s\n" % self.problem.task_type)
        self.logfile.write("Metrics: %s\n" % self.problem.metrics)

        pe = planner_event_handler

        self._show_status("Planning...")

        # Get data details
        df = copy.copy(self.data_manager.input_data)
        df_lbl = copy.copy(self.data_manager.target_data)
        df_profile = DataProfile(df)
        self.logfile.write("Data profile: %s\n" % df_profile)
        l2_pipelines_map = {}
        l1_pipelines_handled = {}
        l2_pipelines_handled = {}

        
        if self.model_name_flag == False:
            l1_pipelines = self.l1_planner.get_pipelines(df)

            if l1_pipelines is None:
                # If no L1 Pipelines, then we don't support this problem
                yield pe.ProblemNotImplemented()
                return
            self.exec_pipelines = []

        #modified by chen, add one more pipeline using the particular model 
        # print(l1_pipelines)
        else:
            models = self.model_name

            if self.feature_extraction_flag == True:
            	feature_extraction = self.feature_extraction
            	print(feature_extraction)
            	l1_pipelines = self.l1_planner.get_particular_pipelines_by_feature_extration(df, models,feature_extraction)
            else:
            	l1_pipelines = self.l1_planner.get_particular_pipelines(df, models)


            if l1_pipelines is None:
                # If no L1 Pipelines, then we don't support this problem
                yield pe.ProblemNotImplemented()
                return
            self.exec_pipelines = []

        if self.feature_extraction_flag == True:
            feature_extraction = self.feature_extraction
            print(feature_extraction)
            l1_pipelines = self.l1_planner.get_pipelines_by_feature_extration(df, feature_extraction)

        if self.exclude_model_flag == True:
            exclude_model = self.exclude_model
            self.exclude_model_function(l1_pipelines, exclude_model)

        if self.replace_model_flag == True:
            replaced_model = self.replace_model.get("replaced_model")
            new_model = self.replace_model.get("new_model")
            print(replaced_model)
            print(new_model)   

            self.same_step_with_replaced_model(l1_pipelines, replaced_model, new_model)         
            


   

    # TODO: Do Pipeline Hyperparameter Tuning

    # Pick top N pipelines, and get similar pipelines to it from the L1 planner to further explore
        print(l1_pipelines)


        while len(l1_pipelines) > 0:
            self.logfile.write("\nL1 Pipelines:\n-------------\n")
            self.logfile.write("%s\n" % str(l1_pipelines))
            self.logfile.write("-------------\n")


            l2_l1_map = {}

            self._show_status("Exploring %d basic pipeline(s)..." % len(l1_pipelines))

            l2_pipelines = []
            for l1_pipeline in l1_pipelines:
                if l1_pipelines_handled.get(str(l1_pipeline), False):
                    continue
                l2_pipeline_list = self.l2_planner.expand_pipeline(l1_pipeline, df_profile)
                l1_pipelines_handled[str(l1_pipeline)] = True
                if l2_pipeline_list:
                    for l2_pipeline in l2_pipeline_list:
                        if not l2_pipelines_handled.get(str(l2_pipeline), False):
                            l2_l1_map[l2_pipeline.id] = l1_pipeline
                            l2_pipelines.append(l2_pipeline)
                            l2_pipelines_map[str(l2_pipeline)] = l2_pipeline
                            yield pe.SubmittedPipeline(l2_pipeline)
            # print(l2_pipelines)

            self.logfile.write("\nL2 Pipelines:\n-------------\n")
            self.logfile.write("%s\n" % str(l2_pipelines))

            self._show_status("Found %d executable pipeline(s). Testing them..." % len(l2_pipelines))

            for l2_pipeline in l2_pipelines:
            	yield pe.RunningPipeline(l2_pipeline)
                # exec_pipeline = self.l2_planner.patch_and_execute_pipeline(l2_pipeline, df, df_lbl)

            if self.imputation_flag == True:
                imputation = self.imputation
                exec_pipelines = self.resource_manager.execute_pipelines_imputation(l2_pipelines, df, df_lbl, imputation)
            else:             
                exec_pipelines = self.resource_manager.execute_pipelines(l2_pipelines, df, df_lbl)

            # exec_pipelines = self.resource_manager.execute_pipelines(l2_pipelines, df, df_lbl)
            for exec_pipeline in exec_pipelines:
                l2_pipeline = l2_pipelines_map[str(exec_pipeline)]
                l2_pipelines_handled[str(l2_pipeline)] = True
                yield pe.CompletedPipeline(l2_pipeline, exec_pipeline)
                if exec_pipeline:
                    self.exec_pipelines.append(exec_pipeline)


            self.exec_pipelines = sorted(self.exec_pipelines, key=lambda x: self._sort_by_metric(x))
            self.logfile.write("\nL2 Executed Pipelines:\n-------------\n")
            self.logfile.write("%s\n" % str(self.exec_pipelines))
            self.write_training_results()

            # print(l1_pipelines)



            if self.model_name_flag == False:
            	l1_related_pipelines = []
            	for index in range(0, cutoff):
            		if index >= len(self.exec_pipelines):
            			break
            			l1_pipeline = l2_l1_map.get(self.exec_pipelines[index].id)
            			if l1_pipeline:
            				related_pipelines = self.l1_planner.get_related_pipelines(l1_pipeline)
            			for related_pipeline in related_pipelines:
            				if not l1_pipelines_handled.get(str(related_pipeline), False):
            					l1_related_pipelines.append(related_pipeline)
            	self.logfile.write("\nRelated L1 Pipelines to top %d L2 Pipelines:\n-------------\n" % cutoff)
            	self.logfile.write("%s\n" % str(l1_related_pipelines))
            	l1_pipelines = l1_related_pipelines

            break
            


    def same_step_with_replaced_model(self, pipeline, replaced_model, new_model):
        new_pipelines = []
        # repaced_pipeline = []
        # pipeline_copy = pipeline.clone()
        for elem in pipeline:
            primitive = elem.getPrimitiveAt(-1)
            for model in replaced_model:
                if model == primitive.name:
                    replaced_pipeline = elem
                    pipeline.remove(replaced_pipeline)
        for model in new_model:
            #make a new primitive
            # print()
            cls = Primitives()
            new_primitive = Primitive(model,cls)
            replaced_pipeline.replacePrimitiveAt(-1, new_primitive)
        return pipeline.append(replaced_pipeline)

    def exclude_model_function(self, pipeline, exclude_model):
        new_pipelines = []
        for elem in pipeline:
            primitive = elem.getPrimitiveAt(-1)
            for model in exclude_model:
                if model == primitive.name:
                    replaced_pipeline = elem
                    pipeline.remove(replaced_pipeline)
        return pipeline


    '''
    Write training results to file
    '''
    def write_training_results(self):
        # Sort pipelines
        self.exec_pipelines = sorted(self.exec_pipelines, key=lambda x: self._sort_by_metric(x))

        # Ended planners
        self._show_status("Found total %d successfully executing pipeline(s)..." % len(self.exec_pipelines))

        # Create executables
        self.pipelinesfile.write("# Pipelines ranked by metrics (%s)\n" % self.problem.metrics)
        for index in range(0, len(self.exec_pipelines)):
            pipeline = self.exec_pipelines[index]
            rank = index + 1
            # Format the metric values
            metric_values = []
            for metric in pipeline.planner_result.metric_values.keys():
                metric_value = pipeline.planner_result.metric_values[metric]
                metric_values.append("%s = %2.4f" % (metric, metric_value))

            self.pipelinesfile.write("%s ( %s ) : %s\n" % (pipeline.id, pipeline, metric_values))
            self.execution_helper.create_pipeline_executable(pipeline, self.config)
            self.create_pipeline_logfile(pipeline, rank)

    '''
    Predict results on test data given a pipeline
    '''
    def test(self, pipeline, test_event_handler):
        helper = ExecutionHelper(self.problem, self.data_manager)
        testdf = pd.DataFrame(copy.copy(self.data_manager.input_data))
        target_col = self.data_manager.target_columns[0]['colName']
        print("** Evaluating pipeline %s" % str(pipeline))
        sys.stdout.flush()
        for primitive in pipeline.primitives:
            # Initialize primitive
            try:
                print("Executing %s" % primitive)
                sys.stdout.flush()
                if primitive.task == "Modeling":
                    result = pd.DataFrame(primitive.executables.predict(testdf), index=testdf.index, columns=[target_col])
                    pipeline.test_result = PipelineExecutionResult(result, None)
                    break
                elif primitive.task == "PreProcessing":
                    testdf = helper.test_execute_primitive(primitive, testdf)
                elif primitive.task == "FeatureExtraction":
                    testdf = helper.test_featurise(primitive, testdf)
                if testdf is None:
                    break
            except Exception as e:
                sys.stderr.write(
                    "ERROR test(%s) : %s\n" % (pipeline, e))
                traceback.print_exc()

        yield test_event_handler.ExecutedPipeline(pipeline)

    def stop(self):
        '''
        Stop planning, and write out the current list (sorted by metric)
        '''

    def create_pipeline_logfile(self, pipeline, rank):
        logfilename = "%s%s%s.json" % (self.log_dir, os.sep, pipeline.id)
        logdata = {
            "problem_id": self.problem.prID,
            "pipeline_rank": rank,
            "name": pipeline.id,
            "primitives": []
        }
        for primitive in pipeline.primitives:
            logdata['primitives'].append(primitive.cls)
        with(open(logfilename, 'w')) as pipelog:
            json.dump(logdata, pipelog,
                sort_keys=True, indent=4, separators=(',', ': '))
            pipelog.close()

    def _dir(self, config, key, makeflag=False):
        dir = config.get(key)
        if dir is None:
            return None
        dir = os.path.abspath(dir)
        if makeflag and not os.path.exists(dir):
            os.makedirs(dir)
        return dir

    def _show_status(self, status):
        print(status)
        sys.stdout.flush()

    def _sort_by_metric(self, pipeline):
        # NOTE: Sorting/Ranking by first metric only
        metric_name = self.problem.metrics[0].name
        mlower = metric_name.lower()
        if "error" in mlower or "loss" in mlower or "time" in mlower:
            return pipeline.planner_result.metric_values[metric_name]
        return -pipeline.planner_result.metric_values[metric_name]
