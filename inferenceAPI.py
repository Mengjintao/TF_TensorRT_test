import time
import os 
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt

class inferenceAPI():
    def __init__(self, path, input_name, output_name, model_name=None, init_dict=None):
        """
        Tensorflow & TensorRT FrozenGraph Loader
        :param path: model path, where the .pb file is
        :param input_name: list of input tensor names
        :param output_name: list of output tensor names
        :param model_name: your given name, None by default
        :param init_dict: None by default, a dict stands for some initialization parameters, like:
            { init_tensor_name_1: init_tensor_value_1, init_tensor_name_2: init_tensor_value_2, ...}
        """
        f = tf.gfile.GFile(path, 'rb')
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        self.trt_graph = trt.create_inference_graph(
            input_graph_def=graph_def,
            outputs=output_name,
            max_batch_size=1,
            max_workspace_size_bytes=8000000000,
            precision_mode="FP16")

        print (input_name)
        print (output_name)
        self.input_name = input_name
        self.output_name = output_name
        self.model_path = path
        self.model_name = model_name

    def inference(self, net_input):
        net_input = tf.convert_to_tensor(net_input,  dtype=tf.float32)
        with tf.Session() as sess:
            output_node = tf.import_graph_def(
                self.trt_graph,
                input_map={self.input_name[0]:net_input},
                return_elements=self.output_name)

            start = time.time()
            res = sess.run(output_node)
            end   = time.time()
            print("Average timeuasge:"),
            print((end-start)*1000)
            return res
