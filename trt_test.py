import numpy as np
import sys
import time
import argparse
import inferenceAPI as   trtAPI

def inference(tf_model1, in_names1, out_names1, tf_model2, in_names2, out_names2, transpose_flag=False):
    print (in_names1)
    print (out_names1)
    in_names_tf1  = [name + ':0' for name in in_names1]
    out_names_tf1 = [name + ':0' for name in out_names1]

    in_names_tf2  = [name + ':0' for name in in_names2]
    out_names_tf2 = [name + ':0' for name in out_names2]

    net_input  = np.random.random_sample((1,2048,2048,3))
    trt_engine1 = trtAPI.inferenceAPI(path=tf_model1, input_name=in_names_tf1, output_name=out_names_tf1, model_name="1")
    trt_engine2 = trtAPI.inferenceAPI(path=tf_model2, input_name=in_names_tf2, output_name=out_names_tf2, model_name="2")

    trt_result1 = trt_engine1.inference(net_input)
    trt_result2 = trt_engine2.inference(net_input)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tensorflow Inference Engine')
    parser.add_argument('--model1', dest='model_path1', type=str, required=True)
    parser.add_argument('--input_node1',  dest='input_node1',  type=str,  required=True)
    parser.add_argument('--output_node1', dest='output_node1', type=str, required=True)

    parser.add_argument('--model2', dest='model_path2', type=str, required=True)
    parser.add_argument('--input_node2',  dest='input_node2',  type=str,  required=True)
    parser.add_argument('--output_node2', dest='output_node2', type=str, required=True)

    args = parser.parse_args()

    inference(args.model_path1, [args.input_node1], [args.output_node1], args.model_path2, [args.input_node2], [args.output_node2])
