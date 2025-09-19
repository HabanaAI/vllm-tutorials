#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import math
import os

import pandas as pd


def get_device_model():
    import habana_frameworks.torch.hpu as hthpu
    os.environ["LOG_LEVEL_ALL"] = "6"
    hpu_determined = hthpu.get_device_name()
    return hpu_determined

def prepare_group_list(grouping_string):
    group_seperator = '--'
    card_seperator = ','
    grouping_string = grouping_string.replace(" ", "").strip(group_seperator).strip(card_seperator)
    group_list=[group.strip(card_seperator).split(card_seperator) for group in grouping_string.split(group_seperator)]
    print("Card grouping list >> {}".format(group_list))
    return group_list

def vllm_auto_calc(fd):
    # handle exceptions
    if (fd['MODEL'] in [
        'Qwen/Qwen2.5-32B-Instruct'
    ] and DTYPE == "fp8"
    and hpu_determined == "GAUDI2"):
        fd['GPU_FREE_MEM_TARGET'] = 3
    
    if DTYPE == "fp8":
        fd['QUANT_DTYPE'] = 1
        fd['CACHE_DTYPE_BYTES'] = fd['CACHE_DTYPE_BYTES_FP8']
        if os.environ.get('TENSOR_PARALLEL_SIZE') is None:
            fd['TENSOR_PARALLEL_SIZE'] = fd['TENSOR_PARALLEL_SIZE_FP8']

    tensor_parallel_size_new = max(1, min(8, fd['TENSOR_PARALLEL_SIZE']))
    if tensor_parallel_size_new != fd['TENSOR_PARALLEL_SIZE']:
        print(f"Clamping TENSOR_PARALLEL_SIZE to {tensor_parallel_size_new}")
    fd['TENSOR_PARALLEL_SIZE'] = tensor_parallel_size_new

    if DTYPE == "fp8":
        if fd['UNI_GROUPS'] == "\"None\"":
            fd['MEASUREMENT_TP'] = fd['TENSOR_PARALLEL_SIZE']
        elif fd['MEASUREMENT_TP'] <= fd['TENSOR_PARALLEL_SIZE']:
            raise ValueError(
                "Unification is not \"None\". "
                "Set correct MEASUREMENT_TP and UNI_GROUPS")
        else:
            groups = prepare_group_list(fd['UNI_GROUPS'])
            mtp = sum(len(cards) for cards in groups)
            if mtp != fd['MEASUREMENT_TP']:
                raise ValueError(
                    f"MEASUREMENT_TP {fd['MEASUREMENT_TP']} not same as total unified cards {mtp}. "
                    "Set correct MEASUREMENT_TP and UNI_GROUPS")
            if len(groups) != fd['TENSOR_PARALLEL_SIZE']:
                raise ValueError(
                    f"TP {fd['TENSOR_PARALLEL_SIZE']} not same than unified groups {len(groups)}. "
                    "Set correct TENSOR_PARALLEL_SIZE, MEASUREMENT_TP and UNI_GROUPS")

    fd['MAX_MODEL_LEN'] = max(1, fd['MAX_MODEL_LEN'])

    if fd['TENSOR_PARALLEL_SIZE'] > 1:
        fd['PT_HPU_ENABLE_LAZY_COLLECTIVES'] = True
    else:
        fd['PT_HPU_ENABLE_LAZY_COLLECTIVES'] = False

    fd['MODEL_MEM_FROM_CONFIG'] = float(fd.get('MODEL_MEM_FROM_CONFIG'))
    fd['DTYPE'] = DTYPE
    fd['DEVICE_HPU_MEM'] = hpu_mem[hpu_determined]

    print(f"{hpu_determined} Device detected with "
          f"{fd['DEVICE_HPU_MEM']} GB memory.")

    fd['TOTAL_GPU_MEM'] = fd['DEVICE_HPU_MEM'] * fd['TENSOR_PARALLEL_SIZE']
    fd['MODEL_MEM_IN_GB'] = (fd['MODEL_MEM_FROM_CONFIG'] * fd['QUANT_DTYPE'] /
                             fd['MODEL_DTYPE']) / (1024 * 1024 * 1024)
    fd['KV_SIZE'] = fd['HEAD_DIM'] * fd['NUM_KEY_VALUE_HEADS'] if fd['HEAD_DIM'] else \
                    fd['HIDDEN_SIZE'] * fd['NUM_KEY_VALUE_HEADS'] / fd['NUM_ATTENTION_HEADS']
    
    aa = fd['MAX_MODEL_LEN'] * fd['NUM_HIDDEN_LAYERS'] * (fd['KV_LORA_RANK'] + fd['QK_ROPE_HEAD_DIM']) * \
        fd['CACHE_DTYPE_BYTES']
    bb = 2 * fd['MAX_MODEL_LEN'] * fd['NUM_HIDDEN_LAYERS'] * fd['KV_SIZE'] * fd['CACHE_DTYPE_BYTES']
    fd['KV_CACHE_PER_SEQ_BYTES'] = aa if fd['KV_LORA_RANK'] else bb
    fd['KV_CACHE_PER_SEQ'] = fd['KV_CACHE_PER_SEQ_BYTES'] / 1024 / 1024 / 1024

    cc = fd['KV_CACHE_PER_SEQ'] * fd['LIMIT_MODEL_LEN'] / fd['MAX_MODEL_LEN']
    dd = min(fd['TENSOR_PARALLEL_SIZE'], fd['NUM_KEY_VALUE_HEADS'])
    fd['MAX_NUM_BATCHED_TOKEN'] = ((cc if fd['SAVE_RECIPE_CACHE'] else fd['KV_CACHE_PER_SEQ']) /
                                    (1 if fd['KV_LORA_RANK'] else dd))
    fd['USABLE_MEM'] = ((fd['TOTAL_GPU_MEM'] / fd['TENSOR_PARALLEL_SIZE']) -
                        fd['UNAVAILABLE_MEM_ABS'] -
                        (fd['MODEL_MEM_IN_GB'] / fd['TENSOR_PARALLEL_SIZE']) -
                        fd['MAX_NUM_BATCHED_TOKEN'])                    
    if fd['USABLE_MEM'] < 0:
        raise ValueError(
            f"Not enough memory for MODEL '{os.environ['MODEL']}', "
            "increase TENSOR_PARALLEL_SIZE.")
    else:
        print(f"Usable graph+kvcache memory {fd['USABLE_MEM']:.2f} GB")

    if fd.get('GPU_MEMORY_UTILIZATION') is None:
        fd['GPU_MEMORY_UTILIZATION'] = math.floor(
                                        (1 - fd['GPU_FREE_MEM_TARGET'] / fd['USABLE_MEM']) 
                                        * 100) / 100
    fd['KVCACHE_MEM_EST'] = fd['USABLE_MEM'] * fd['GPU_MEMORY_UTILIZATION']
    fd['KVCACHE_PARALLEL'] = 1 if fd['KV_LORA_RANK'] else min(fd['TENSOR_PARALLEL_SIZE'], fd['NUM_KEY_VALUE_HEADS']) 
    if fd.get('MAX_NUM_SEQS') is None:
        fd['EST_MAX_NUM_SEQS'] = fd['MAX_NUM_SEQS_CONFIG'] if fd['MAX_NUM_SEQS_CONFIG'] else \
                                fd['KVCACHE_MEM_EST'] / (fd['KV_CACHE_PER_SEQ'] / fd['KVCACHE_PARALLEL'])
    else:
        fd['EST_MAX_NUM_SEQS'] = max(1, fd['MAX_NUM_SEQS'])

    if fd['EST_MAX_NUM_SEQS'] < 1:
        raise ValueError(
            "Not enough memory for kv cache. "
            "Increase TENSOR_PARALLEL_SIZE or reduce MAX_MODEL_LEN")
    print(f"Estimating graph memory for "
          f"{fd['EST_MAX_NUM_SEQS']:.2f} MAX_NUM_SEQS")

    if fd.get('NUM_GPU_BLOCKS_OVERRIDE') is None:
        fd['EST_HPU_BLOCKS'] = int((fd['MAX_MODEL_LEN'] * fd['KVCACHE_MEM_EST']) / (fd['KV_CACHE_PER_SEQ'] / 
                                    fd['KVCACHE_PARALLEL']) / fd['BLOCK_SIZE'])
        fd['NUM_GPU_BLOCKS_OVERRIDE'] = int(fd['EST_HPU_BLOCKS'])
    else:
        fd['EST_HPU_BLOCKS'] = fd['NUM_GPU_BLOCKS_OVERRIDE']

    ee = 1 + math.ceil(math.log(fd['EST_MAX_NUM_SEQS'], 2))
    ff = 1 + int(math.log(
                        fd['VLLM_DECODE_BS_BUCKET_STEP'] / fd['VLLM_DECODE_BS_BUCKET_MIN'], 
                        2))
    fd['DECODE_BS_RAMP_GRAPHS'] = ee if fd['VLLM_EXPONENTIAL_BUCKETING'] else ff

    gg = max(0, int(1 + (fd['EST_MAX_NUM_SEQS'] - fd['VLLM_DECODE_BS_BUCKET_STEP']) / 
            fd['VLLM_DECODE_BS_BUCKET_STEP']))
    fd['DECODE_BS_STEP_GRAPHS'] = 0 if fd['VLLM_EXPONENTIAL_BUCKETING'] else gg

    hh = 1 + math.ceil(math.log(fd['EST_HPU_BLOCKS'], 2))
    ii = 1 + int(math.log(
                        fd['VLLM_DECODE_BLOCK_BUCKET_STEP'] / fd['VLLM_DECODE_BLOCK_BUCKET_MIN'], 
                        2))
    fd['DECODE_BLOCK_RAMP_GRAPHS'] = hh if fd['VLLM_EXPONENTIAL_BUCKETING'] else ii

    jj = max(0, int(1 + 
                    (fd['EST_HPU_BLOCKS'] - fd['VLLM_DECODE_BLOCK_BUCKET_STEP']) / 
                    fd['VLLM_DECODE_BLOCK_BUCKET_STEP']))
    fd['DECODE_BLOCK_STEP_GRAPHS'] = 0 if fd['VLLM_EXPONENTIAL_BUCKETING'] else jj
    
    fd['NUM_DECODE_GRAPHS'] = (
        (fd['DECODE_BS_RAMP_GRAPHS'] + fd['DECODE_BS_STEP_GRAPHS']) *
        (fd['DECODE_BLOCK_RAMP_GRAPHS'] + fd['DECODE_BLOCK_STEP_GRAPHS']))

    kk = 1 + math.ceil(math.log(fd['MAX_NUM_PREFILL_SEQS'], 2))
    ll = 1 + int(math.log(
                        min(fd['MAX_NUM_PREFILL_SEQS'], fd['VLLM_PROMPT_BS_BUCKET_STEP']) / 
                            fd['VLLM_PROMPT_BS_BUCKET_MIN'], 
                            2))
    fd['PROMPT_BS_RAMP_GRAPHS'] = kk if fd['VLLM_EXPONENTIAL_BUCKETING'] else ll 

    mm =  max(0, int(1 + 
                        (fd['MAX_NUM_PREFILL_SEQS'] - fd['VLLM_PROMPT_BS_BUCKET_STEP']) / 
                        fd['VLLM_PROMPT_BS_BUCKET_STEP']))
    fd['PROMPT_BS_STEP_GRAPHS'] = 0 if fd['VLLM_EXPONENTIAL_BUCKETING'] else mm

    nn = 1 + math.ceil(math.log(
                                fd['MAX_MODEL_LEN'], 
                                2))
    oo = 1 + int(math.log(
                        fd['VLLM_PROMPT_SEQ_BUCKET_STEP'] / 
                        fd['VLLM_PROMPT_SEQ_BUCKET_MIN'], 
                        2))
    fd['PROMPT_SEQ_RAMP_GRAPHS'] = nn if fd['VLLM_EXPONENTIAL_BUCKETING'] else oo

    pp = int(1 + ((fd['MAX_MODEL_LEN'] - fd['VLLM_PROMPT_SEQ_BUCKET_STEP']) / 
                    fd['VLLM_PROMPT_SEQ_BUCKET_STEP']))
    fd['PROMPT_SEQ_STEP_GRAPHS'] = 0 if fd['VLLM_EXPONENTIAL_BUCKETING'] else pp
    
    fd['EST_NUM_PROMPT_GRAPHS'] = (
        (fd['PROMPT_BS_RAMP_GRAPHS'] + fd['PROMPT_BS_STEP_GRAPHS']) *
        (fd['PROMPT_SEQ_RAMP_GRAPHS'] + fd['PROMPT_SEQ_STEP_GRAPHS']) / 2)
    fd['EST_GRAPH_PROMPT_RATIO'] = math.ceil(
        fd['EST_NUM_PROMPT_GRAPHS'] /
        (fd['EST_NUM_PROMPT_GRAPHS'] + fd['NUM_DECODE_GRAPHS']) * 100) / 100
    print(f"Estimated Prompt graphs {fd['EST_NUM_PROMPT_GRAPHS']:.0f} and "
          f"Decode graphs {fd['NUM_DECODE_GRAPHS']}")
    fd['VLLM_GRAPH_PROMPT_RATIO'] = math.ceil(
        min(max(fd['EST_GRAPH_PROMPT_RATIO'], 0.1), 0.9) * 10) / 10
    fd['DECODE_GRAPH_TARGET_GB'] = math.ceil(
        fd['NUM_DECODE_GRAPHS'] * fd['APPROX_MEM_PER_GRAPH_MB'] / 1024 *
        10) / 10
    fd['EST_GRAPH_RESERVE_MEM'] = math.ceil(
        fd['DECODE_GRAPH_TARGET_GB'] /
        (fd['USABLE_MEM'] * fd['GPU_MEMORY_UTILIZATION'] *
         (1 - fd['VLLM_GRAPH_PROMPT_RATIO'])) * 100) / 100
    fd['VLLM_GRAPH_RESERVED_MEM'] = min(max(fd['EST_GRAPH_RESERVE_MEM'], 0.01),
                                        0.5)
    fd['KV_CACHE_MEM'] = (fd['USABLE_MEM'] * fd['GPU_MEMORY_UTILIZATION'] *
                          (1 - fd['VLLM_GRAPH_RESERVED_MEM']))

    if fd.get('MAX_NUM_SEQS') is None:
        rr = fd['KV_CACHE_MEM'] / (fd['KV_CACHE_PER_SEQ'] / 
                                    (1 if fd['KV_LORA_RANK'] else 
                                    min(fd['TENSOR_PARALLEL_SIZE'], fd['NUM_KEY_VALUE_HEADS'])
                                    )
                                    )
        fd['MAX_NUM_SEQS'] = fd['MAX_NUM_SEQS_CONFIG'] if fd['MAX_NUM_SEQS_CONFIG'] else rr
        if DTYPE == 'fp8':
            fd['MAX_NUM_SEQS'] = (max(
                1,
                math.floor(
                    fd['MAX_NUM_SEQS'] / fd['VLLM_DECODE_BS_BUCKET_STEP']),
            ) * fd['VLLM_DECODE_BS_BUCKET_STEP'])
        else:
            fd['MAX_NUM_SEQS'] = (math.ceil(
                fd['MAX_NUM_SEQS'] / fd['VLLM_DECODE_BS_BUCKET_STEP']) *
                                  fd['VLLM_DECODE_BS_BUCKET_STEP'])

        if fd['MAX_NUM_SEQS'] < 1:
            raise ValueError(
                "Not enough memory for kv cache increase TENSOR_PARALLEL_SIZE "
                "or reduce MAX_MODEL_LEN or increase bucket step")

        if (fd['MODEL'] in [
                'meta-llama/Llama-3.2-11B-Vision-Instruct',
                'meta-llama/Llama-3.2-90B-Vision-Instruct'
        ] and fd['MAX_NUM_SEQS'] > 128):
            fd['MAX_NUM_SEQS'] = 128
            print(f"{fd['MODEL']} currently does not support "
                  "max-num-seqs > 128. "
                  "Limiting max-num-seqs to 128")
        print("Setting MAX_NUM_SEQS", fd['MAX_NUM_SEQS'])
    else:
        fd['MAX_NUM_SEQS'] = max(1, fd['MAX_NUM_SEQS'])

    fd['VLLM_DECODE_BLOCK_BUCKET_MAX'] = max(
        128, math.ceil((fd['MAX_NUM_SEQS'] * fd['MAX_MODEL_LEN']) / 128))
    fd['VLLM_PROMPT_SEQ_BUCKET_MAX'] = fd['MAX_MODEL_LEN']

    if hpu_determined == "GAUDI2":
        fd["gnum"]='g2'
    elif hpu_determined == "GAUDI3":
        fd["gnum"]='g3'

    # DEBUG ******************************
    df = pd.DataFrame(list(fd.items()), columns=['Param', 'Value'])
    pd.set_option('display.max_rows', None)
    print(df)
    # ************************************

    # Create our output list
    with open('varlist_output.txt') as ovp_file:
        for line in ovp_file:
            param = line.strip()
            output_dict[param] = fd[param]

    # Append user updatable list
    with open('varlist_userupd.txt') as ovp_file:
        for line in ovp_file:
            param = line.strip()
            if os.environ.get(param) is not None:
                output_dict[param] = fd[param]
    return output_dict


def get_model_from_csv(file_path):
    # Read settings CSV and return dict
    dataframe_csv = pd.read_csv(file_path)
    filtered_row = dataframe_csv.loc[dataframe_csv['MODEL'] ==
                                     os.environ['MODEL']]

    if filtered_row.empty:
        raise ValueError(
            f"No matching rows found for MODEL '{os.environ['MODEL']}' "
            f"in {file_path}")

    # CSV should not have more than 1 row for each MODEL.
    # But just in case, return the first
    try:
        filtered_dict = filtered_row.to_dict(orient='records')[0]
    except Exception as err:
        raise ValueError(
            "Unsupported MODEL or MODEL not defined! Exiting.") from err

    return filtered_dict


def overwrite_params(dict_before_updates):
    # Overwrite default values with user provided ones before auto_calc
    with open('varlist_userupd.txt') as ovp_file:
        for line in ovp_file:
            param = line.strip()
            if os.environ.get(param) is not None:
                try:
                    dict_before_updates[param] = eval(os.environ[param])
                except Exception:
                    dict_before_updates[param] = os.environ[param]

                print(f"Adding or updating {param} "
                      f"to {dict_before_updates[param]}")

    return dict_before_updates


def write_dict_to_file(fd, file):
    with open(file, 'w') as file_obj:
        for key, value in fd.items():
            file_obj.write(f"export {key}={value}\n")


def main():
    global hpu_mem, hpu_determined, DTYPE, output_dict

    # CONSTANTS
    hpu_mem = {'GAUDI2': 96, 'GAUDI3': 128}

    if os.getenv('DTYPE') is None:
        DTYPE = 'bfloat16'
    else:
        DTYPE = os.environ['DTYPE']

    # PRECHECKS
    if os.getenv('MODEL') is None:
        print('Could not determine which model to use. '
              'Provide a model name in env-var "MODEL"')
        exit(-1)

    # Output vars
    file_input_csv = 'settings_vllm.csv'
    file_output_vars = 'server_vars.txt'
    output_dict = {}

    # Get HPU model and filter row by HPU again
    hpu_determined = get_device_model()

    # Read settings csv into a dataframe
    try:
        fd = get_model_from_csv(file_input_csv)
    except ValueError as e:
        print("Error:", e)
        exit(-1)

    # Use a single if statement for MAX_MODEL_LEN
    if (os.getenv('MAX_MODEL_LEN') is not None
            and int(os.environ['MAX_MODEL_LEN']) > fd['LIMIT_MODEL_LEN']):
        print(f"Supplied MAX_MODEL_LEN {os.environ['MAX_MODEL_LEN']} "
              "cannot be higher than the permissible value "
              f"{str(fd['LIMIT_MODEL_LEN'])} for this MODEL.")
        exit(-1)

    # Overwrite params then perform autocalc
    fd = overwrite_params(fd)
    try:
        if fd['MAX_MODEL_LEN_CONFIG'] != 0:
            print('\nRecipe Calc - 1st Iteration *******************')
            fd['MAX_MODEL_LEN'] = fd['MAX_MODEL_LEN_CONFIG']
            fd['MAX_NUM_SEQS_P90_CONFIG'] = 0
            output_dict = vllm_auto_calc(fd)
            print('\nRecipe Calc - 2nd Iteration *******************')
            fd = get_model_from_csv(file_input_csv)
            fd = overwrite_params(fd)
            fd['MAX_NUM_SEQS_P90_CONFIG'] = output_dict['MAX_NUM_SEQS']
            output_dict = vllm_auto_calc(fd)
        else:
            print('\nRecipe Calc - 1st Iteration *******************')
            output_dict = vllm_auto_calc(fd)

    except ValueError as e:
        print("Error:", e)
        exit(-1)

    # Write to a text file
    write_dict_to_file(output_dict, file_output_vars)


if __name__ == '__main__':
    main()
