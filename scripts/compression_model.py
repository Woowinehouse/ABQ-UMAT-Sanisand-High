# -* - coding:UTF-8 -*-
import os
import sys
from typing import List, Union
# Add the script directory to sys.path so that 'model' module can be found
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
# Also add the parent directory in case script is run from different location
parent_dir = os.path.dirname(script_dir)
# Add absolute path to scripts directory
scripts_abs_path = r"F:/VScode2026/umat_cpp/scripts"
sys.path.insert(0, script_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, scripts_abs_path)
from model import create_model, modify_file
import time


def create_model_and_inp_umat(
    condition: List[List[Union[int, float]]],
    model_path: str,
    input_path : str,
    # umat_name_pre: str,
    # umat_path: str,
    model_name: str,
) -> None:
    """
    基于 abaqus python 接口，通过 python 文件批量生成不同工况的 abaqus 的 .cae 和 .inp 文件，并存放至相应的文件夹当中。
    注意:使用的abaqus 2023的版本,abaqus 内置python版本为python2.7
    :param model_name:
    :param rela_density:
    :param umat_path:
    :param umat_name_pre:
    :param condition: 指定数值仿真的工况
    :param path_test: 保存
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Check whether the folder exists. if it does not, create a new folder to hold the abaqus model
    model_path_exists = os.path.exists(model_path)
    if not model_path_exists:
        os.makedirs(model_path)
        print("model Folder has been created successfully" "")
    else:
        print("model folder already exists")
    # ------------------------------------------------------------------------------------------------------------------
    # create a new folder in which the file contains input file and umat.for
    for P, b, Dr in condition:
        condition_suf = "P{P_name}_b{b_name}_Dr{Dr_name}".format(
            P_name=P, b_name=b, Dr_name=Dr
        )
        inp_name = model_name + "_" + condition_suf
        cae_name = inp_name
        # create a test folder path and job name and appoint job path
        subtest_name = "test_p{P_name}_b{b_name}_Dr{Dr_name}".format(
            P_name=P, b_name=b, Dr_name=Dr
        )
        path_subtest = input_path + "\\" + subtest_name
        #
        # judge whether the folder has been created
        subtest_path_exists = os.path.exists(path_subtest)
        if not subtest_path_exists:
            os.makedirs(path_subtest)
            print("model Folder has been created successfully")
        else:
            print("Folder already exists")
        #
        # create drained cell model and input file
        create_model.drained_cube_model_with_umat(
            P, model_path, path_subtest, inp_name, cae_name
        )
        #
        modify_file.input_inp(path_subtest, inp_name)


def main():
    module_path = "F:/VScode2026/umat_cpp/model"
    input_path  = "F:/VScode2026/umat_cpp/input"
    p_condition = [100]
    b_condition = [0]
    Dr_condition = [60]
    model_name = "drained"
    #
    task_list: List[List[Union[int, float]]] = []
    for P in p_condition:
        for b in b_condition:
            for Dr in Dr_condition:
                task_list.append([P, b, Dr])
    create_model_and_inp_umat(task_list, module_path, input_path, model_name)


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("run_command ", end - start, " seconds")
