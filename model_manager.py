'''This file implements an object to deal the model managemet: the I/O, the saving of the
metadata assosciated, and everything that is helpful during experiments'''

import os
import shutil
import torch
import pickle
from collections import namedtuple
import numbers
import helpers.functions as mhf

template_info_models = namedtuple('saved_model_info_tuple', ('path_saved_model', 'path_saved_metadata'))

USE_CUDA = torch.cuda.is_available()

class ModelManager(object):
    def __init__(self, save_file, name=None, verbose=True, create_new_model_manager=False):

        self.save_file = save_file

        if create_new_model_manager is False:
            try:
                self.load()
                if verbose:
                    print('Model manager "{}" succefully loaded from "{}"'.format(name, save_file))
                return
            except Exception as e:
                raise ValueError('Unable to load model from "{}". Exception: {}'.format(save_file, e))
        else:
            if os.path.exists(save_file):
                raise ValueError('The file specified "{}" already exists. '
                                 'Choose another one'.format(save_file))

            if not isinstance(name, str):
                raise ValueError('"name" parameter must be a string')

            self.name = name
            self.saved_models = {} # the structure is: {model_name: [info_first_run, info_second_run, ... ]}
                                   # where info_first_run are namedtuples with the template specified in
                                   # "template_info_models" variable defined at the beginning of the file
            self.verbose = verbose
            self.save()
            if verbose:
                print('New model "{}" saved at "{}"'.format(name, save_file))

    def __str__(self):
        stringToReturn = 'Model manager "{}" saved at "{}".\n'.format(self.name, self.save_file)
        stringToReturn += 'List of saved models:\n' + \
                         '\n'.join('"{}" saved at ==> "{}"'.format(x, y[0].path_saved_model)
                         for x, y in self.saved_models.items())
        return stringToReturn

    def get_model_base_path(self, model_name):

        if not isinstance(model_name, str):
            raise ValueError('model_name parameter must be a string')

        if model_name not in self.saved_models:
            raise ValueError('The model "{}" is not present in the list of models saved'.format(model_name))

        return self.saved_models[model_name][0].path_saved_model


    def train_model(self, model, model_name, train_function, arguments_train_function, train_loader,
                             test_loader, continue_training_from=-1, continued_training_model_name=None,
                             base_path_continued_training_model=None):

        '''

        :param model: the model instance to train 准备训练的模型
        :param model_name: the model name; must be present in the list of saved models already已经存在save_model之中的训练模型
        :param train_function: function to use to train. It will be passed model, train and test loader, and
                               arguments_train_function 训练模型所使用的函数
        :param arguments_train_function: addtional arguments (in the form of a dictionary) that will be passed to the
                                         train function 希望传给这个训练模型的相关参数
        :param train_loader: train loader to be passed to the train function 训练加载器
        :param test_loader: (optional) If not None, it will be passed to the train function, otherwise not 测试test所用的数据加载器
        :param continue_training_from: Specify from which version of model_name you should resume training (included)
                                       Default is the last saved model. 指定需要使用哪一个相关的模型训练中间过程版本，默认缺省是最后一次训练出来的结果
                                       规定不能取比-1小的数字
        :param continued_training_model_name: If None, any history past the *continue_training_from* parameter
                                              will be overwritten (i.e. we continue training from a particular
                                              checkpoint and discard other successive training runs). If not None,
                                              then the history of model_name will be copied up until
                                              *continue_training_from* and the new iteration will be saved
                                              with the new name *continued_training_model_name*
                                              如果该参数是None的话就会将所有相关的训练记录都给清除重写，否则的话会另外将记录保存
        :param base_path_continued_training_model: Only used if *continued_training_model_name* is not None. It
                                                   specifies where to save the continued model.
                                                   结合上面的参数，如果想另起炉灶重新训练的话就会需要制指定相关的参数（保存新模型吧的保存路径）
        :return:
        '''


        if not isinstance(model_name, str):
            raise ValueError('"model_name" parameter must be a string')

        if not isinstance(continue_training_from, int):
            raise ValueError('"continue_training_from" parameter must be a int')
        # add_model的时候已经将相应的信息进行了注册
        if model_name not in self.saved_models:
            raise ValueError('the model_name specified ({}) does not exist in the list ' 
                                               'of saved models'.format(model_name))

        if continued_training_model_name is not None:
            if base_path_continued_training_model is None:
                raise ValueError('If you want to save the result under the new model name "{}" you need to specify'
                                 ' a new model path.')
            elif os.path.exists(base_path_continued_training_model):
                raise ValueError('The path specified "{}" already exists. '
                                 'Choose a new one'.format(base_path_continued_training_model))

        if continue_training_from < -1:
            raise  ValueError('Parameter continue_training_from must >= -1. -1 indicates the '
                              'last value and is the default')


        num_saved_models = len(self.saved_models[model_name])
        if continue_training_from == -1:
            # meaning you continue from the last trained model
            continue_training_from = num_saved_models - 1 #len-1 相当于是最后一次的运行结果

        #if new_model_name is provided, this trained model will be saved somewhere else with the new name
        #otherwise the history will be rewritten in the same model
        if isinstance(continued_training_model_name, str):
            if continued_training_model_name in self.saved_models:
                raise ValueError('The new model name "{}" is already present. '
                                 'Choose a new one'.format(continued_training_model_name))

            self.saved_models[continued_training_model_name] = self.saved_models[model_name]
            for idx, obj in enumerate(self.saved_models[model_name]):
                #copy the files of the shared history under the new paths
                if idx > continue_training_from:
                    #only copy up to the shared history
                    break
                path_model, path_metadata = obj
                new_path = base_path_continued_training_model + ('' if idx == 0 else str(idx))
                new_path_meta = new_path + '_metadata'
                shutil.copyfile(path_model, new_path)
                shutil.copyfile(path_metadata, new_path_meta)
                self.saved_models[continued_training_model_name][idx] = template_info_models(new_path, new_path_meta)

            model_name = continued_training_model_name

        print('Training model "{}"'.format(model_name))

        #load the wanted model weights and continue
        if continue_training_from > 0:
            #if continue_training_from == 0 then the model has never been trained before（add之后会导致len=1，前面减了1就会导致这个参数的最后结果是0）
            # 注意此时的函数 continue_training_from 值已经不是传入的参数了，上面对这个值进行了修改
            #(it has just been added as a new model) so we load only when continue_training_from > 0
            print('Loading weights from train run number {}'.format(continue_training_from))
            state_dict = torch.load(self.saved_models[model_name][continue_training_from].path_saved_model)
            try:
                model.load_state_dict(state_dict)
            except KeyError:
                #this means that the weight were saved with a DataParallel but the current one is not
                #or vice-versa.
                if isinstance(model, torch.nn.parallel.DataParallel):
                    state_dict = mhf.convert_state_dict_to_data_parallel(state_dict)
                else:
                    state_dict = mhf.convert_state_dict_from_data_parallel(state_dict)
                model.load_state_dict(state_dict)
        else:
            print('Warning: The model will be trained *as is*, no previous weights will be loaded')
        # 看是不是loader能否加载test
        if test_loader is None:
            _, infoDict = train_function(model, train_loader=train_loader, **arguments_train_function)
        else:
            _, infoDict = train_function(model, train_loader=train_loader, test_loader=test_loader,
                                      **arguments_train_function)

        if infoDict.get('numEpochsTrained', 0) == 0:
            print('The run aborted before one epoch is complete. No results will be saved')
            return

        #here we delete the history after continue_after_training
        #first we delete the now useless files
        files_to_delete = [file_name for info_tuple in \
                        self.saved_models[model_name][continue_training_from+1:] for file_name in info_tuple]
        infoMsgRemove = mhf.remove_files_list(files_to_delete)
        if infoMsgRemove and self.verbose: print(infoMsgRemove)
        #and here we delete the dictionary entries
        self.saved_models[model_name] = self.saved_models[model_name][:continue_training_from + 1]

        infoDict['train_function_used'] = 'Function {} defined in {}'.format(train_function.__name__,
                                                                             train_function.__code__.co_filename)
        path_to_save_model = self.saved_models[model_name][0].path_saved_model + str(continue_training_from+1)
        torch.save(model.state_dict(), path_to_save_model)
        path_to_save_metadata = path_to_save_model + '_metadata'
        self.save_metadata((arguments_train_function, infoDict), path_to_save_metadata)
        self.saved_models[model_name].append(template_info_models(path_to_save_model, path_to_save_metadata))

        self.save()

    def add_new_model(self, model_name, path_to_save, arguments_creator_function=None):
        # prequires: model_name不允许已经出现在Model——list中预期希望存储model的filename不能已经存在
        # effects： 出现在指定的path下出现了两个文件，一个是model，一个是Model_metadata
        #           swlf.saved_models中增加了相应的model_name

        if arguments_creator_function is None: arguments_creator_function = {}

        if not isinstance(model_name, str):
            raise ValueError('model_name parameter must be a string')
        # add model的时候不允许这个model已经在list之中
        if model_name in self.saved_models:
            raise ValueError('The model name "{}" is already present. Choose a new name'.format(model_name))
        # add model的时候也不允许存储model预计存储路径已经存在了
        if os.path.exists(path_to_save):
            raise ValueError('The path specified "{}" already exists. Choose a new one'.format(path_to_save))

        #create the empty file for consistency
        with open(path_to_save, 'wb') as _: pass
        path_to_save_metadata = path_to_save + '_metadata'
        infoDict = {}
        self.save_metadata((arguments_creator_function, infoDict), path_to_save_metadata)
        self.saved_models[model_name] = [template_info_models(path_to_save, path_to_save_metadata)]

        self.save()

    def save_metadata(self, obj_to_save, path):

        '''

        obj_to_save should be a list (or tuple) of dictionaries.
        This function makes sure what we save is pickable.

       '''

        good_types = (numbers.Number, str, bool)
        def is_good_type(obj):
            if isinstance(obj, (tuple, list)):
                return all(is_good_type(x) for x in obj)
            return isinstance(obj, good_types) or obj is None

        pickable_object = [{} for _ in obj_to_save]

        for idx, obj in enumerate(obj_to_save):
            if not isinstance(obj, dict):
                raise ValueError('Wrong type: the metadata to save should be a tuple of dictionaries')

            for key, val in obj.items():
                if is_good_type(val):
                    #do nothing
                    pass
                elif callable(val): #function, or class instances that define __call__
                    try:
                        val = 'Name: {}. Repr: {}'.format(val.__name__ , repr(val))
                    except:
                        val = repr(val)
                else:
                    val = repr(val)
                pickable_object[idx][key] = val

        with open(path, 'wb') as p:
            pickle.dump(pickable_object, p)

    def load_metadata(self, model_name, idx_run=-1):

        if not isinstance(model_name, str):
            raise ValueError('model_name parameter must be a string')

        if model_name not in self.saved_models:
            raise ValueError('The model "{}" is not present in the list of models saved'.format(model_name))

        try:
            # 默认情况下会加载最新的model（idx_run=-1代表了最后一次的结果）
            path_metadata = self.saved_models[model_name][idx_run].path_saved_metadata
        except IndexError:
            raise IndexError('There are only {} training runs, but the index passed is {}'.format(
                                                len(self.saved_models[model_name])-1, idx_run))
        with open(path_metadata, 'rb') as p:
            metadata = pickle.load(p)

        return metadata

    def remove_model(self, model_name, delete_files=True):

        "removes the model from the saved models list. If delete_files is False, the actual files won't be deleted"

        infoMsg = ''
        if model_name in self.saved_models:
            if delete_files:
                files_to_delete = [file_name for info_tuple in self.saved_models[model_name] \
                                                                        for file_name in info_tuple]
                infoMsg += mhf.remove_files_list(files_to_delete)
            self.saved_models.pop(model_name, None)
            infoMsg += 'Entry "{}" removed\n'.format(model_name)
        else:
            infoMsg += 'Model ”{}" already not present in memory\n'.format(model_name)

        if self.verbose:
            print(infoMsg)

        self.save()

    def remove_training_runs(self, model_name, idx_run, delete_files=True):

        'removes all runs after idx_run (included) from model history'

        infoMsg = ''
        if model_name not in self.saved_models:
            if self.verbose:
                print('Model ”{}" already not present in memory\n'.format(model_name))
                return

        if (len(self.saved_models[model_name]) - 1) < idx_run:
            if self.verbose:
                print('Model "{}" has less than {} training runs'.format(model_name, idx_run))

        # here we delete the history after continue_after_training
        # first we delete the files
        if delete_files:
            files_to_delete = [file_name for info_tuple in \
                               self.saved_models[model_name][idx_run:] for file_name in info_tuple]
            infoMsg += mhf.remove_files_list(files_to_delete)

        # and here we delete the dictionary entries
        self.saved_models[model_name] = self.saved_models[model_name][:idx_run]

        infoMsg += 'Training runs after {} included were deleted for model "{}"'.format(idx_run, model_name)

        if self.verbose:
            print(infoMsg)

        self.save()


    def save(self):

    # 注意函数的作用是更新manager对应的tst文件
        'saves the model manager to file'
        # 分析：首先save_models是字典：从str到一个列表，故val是列表，[tuple(x) for x in val]现将列表中每个元素取出来变成元组，
        # 然后将元组变成列表，相当于把列表中的元素（每一次运行info）变成元组形式????   用意何在？？？？？
        saved_models_tuple = {key: [tuple(x) for x in val] for key, val in self.saved_models.items()}
        saving_object = (self.name, self.verbose, saved_models_tuple)

        with open(self.save_file+'temp', 'wb') as sf:
            pickle.dump(saving_object, sf)

        #完成一次重命名操作，将相关的数据进行重新写入文件并且重命名
        #now we can remove the old one and rename the temp file. This should prevent data loss
        try:os.remove(self.save_file)
        except:pass

        #this can raise an error if self.save_file was not properly removed
        os.rename(self.save_file+'temp', self.save_file)

    def load(self):

        'loads the model manager from the save_file'
        # tst文件在之前如果有相关的保存记录的时候就会从以前保存下来的对象信息进行加载，而不重新生成一个文件
        with open(self.save_file, 'rb') as sf:
            name, verbose, saved_models = pickle.load(sf)
        self.name = name
        self.verbose = verbose
        #pickle can't pickle named tuples. So when saving we save normal tuples, and when loading we
        #transform it back
        self.saved_models = {key:[template_info_models(*x) for x in val] for key, val in saved_models.items()}

    def load_model_state_dict(self, model_name, idx_run=-1):

        if not isinstance(model_name, str):
            raise ValueError('model_name parameter must be a string')

        if model_name not in self.saved_models:
            raise ValueError('The model "{}" is not present in the list of models saved'.format(model_name))

        if len(self.saved_models[model_name]) - 1 < 1:
            raise ValueError("The model specified hasn't been trained yet")

        try:
            path_saved_model = self.saved_models[model_name][idx_run].path_saved_model
        except IndexError:
            raise IndexError('There are only {} training runs, but the index passed is {}'.format(
                                                len(self.saved_models[model_name])-1, idx_run))
        return torch.load(path_saved_model)

    def list_models(self):
        return list(self.saved_models.keys())

    def get_num_training_runs(self, model_name):
        # requires：输入的model_name必须是字符串类型
        # effects: 最后输出的是相应的len(saved_models[model_name]) - 1
        if not isinstance(model_name, str):
            raise ValueError('model_name parameter must be a string')

        if model_name not in self.saved_models:
            raise ValueError('The model "{}" is not present in the list of models saved'.format(model_name))

        return len(self.saved_models[model_name]) - 1

    def change_model_name(self, old_model_name, new_model_name, change_filenames=True):

        raise NotImplementedError('almost done! Just need to go through this with a debugger ;)')

        # TODO: Make sure the new_model_name does not exist and, if changing filanames, that any of the
        # new filename do not exist before changing anything

        if not (isinstance(old_model_name, str) and isinstance(new_model_name, str)):
            raise ValueError('model_name parameters must be strings')

        if old_model_name not in self.saved_models:
            raise ValueError('The model "{}" is not present in the list of models saved'.format(old_model_name))

        if change_filenames is True:
            #make sure all the filenames to be changed have the current model name as part of their path
            bad_flag = False
            for model_info in self.saved_models[old_model_name]:
                old_model_path = model_info.path_saved_model
                old_model_meta = model_info.path_saved_metadata

                if not (old_model_name in os.path.basename(old_model_path) and
                                old_model_name in os.path.basename(old_model_meta)):
                    bad_flag = True
            if bad_flag:
                raise ValueError('Cannot change the filepath as it does not include the name of the model. '
                                 'Call this function with change_filenames=False to avoid this, but it will '
                                 'be confusing having a certain model name with a filepath completely different')

        if change_filenames is True:
            self.saved_models[new_model_name] = []
            for model_info in self.saved_models[old_model_name]:
                old_model_path = model_info.path_saved_model
                old_model_meta = model_info.path_saved_metadata
                #TODO: bring back the dirpath with os.path.join...
                new_model_path = os.path.basename(old_model_meta).replace(old_model_name, new_model_name)
                new_model_meta = os.path.basename(old_model_meta).replace(old_model_name, new_model_name)
                os.rename(old_model_path, new_model_path)
                os.rename(old_model_meta, new_model_meta)
                self.saved_models[new_model_name].append(template_info_models(new_model_path, new_model_meta))
        else:
            self.saved_models[new_model_name] = self.saved_models[old_model_name]

        self.saved_models.pop(old_model_name, None)
        self.save()

        print('Model name {} changed to {}'.format(old_model_name, new_model_name))

    def change_all_base_paths(self, old_base_path, new_base_path):

        'It will change all paths that start with *old_base_path* and replace them with *new_base_path* instead'

        #TODO: This string based manipulation is probably not robust. Better idea would be to use pathlib.Path
        count_modified = 0
        for model_name in self.saved_models:
            new_model_info = []
            flag_modified_path = False
            for model_info in self.saved_models[model_name]:
                path_lists = [model_info.path_saved_model, model_info.path_saved_metadata]
                for idx, path in enumerate(path_lists):
                    if path.startswith(old_base_path):
                        flag_modified_path = True
                        path_lists[idx] = new_base_path + path[len(old_base_path):]
                new_model_info.append(template_info_models(*path_lists))
            if flag_modified_path:
                count_modified += 1
            self.saved_models[model_name] = new_model_info

        self.save()

        if self.verbose:
            print('The path of {} models has been modified'.format(count_modified))







