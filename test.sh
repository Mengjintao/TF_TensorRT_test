python3 trt_test.py --model1=resnetV150_frozen.pb --input_node1 'input' --output_node1 'resnet_v1_50/predictions/Reshape_1' --model2=model2.pb --input_node2 'input_img' --output_node2 'output_1'
python3 trt_test.py --model1=model1.pb --input_node1 'input_img' --output_node1 'output_1' --model2=model1.pb --input_node2 'input_img' --output_node2 'output_1'
python3 trt_test.py --model1=model1.pb --input_node1 'input_img' --output_node1 'output_1' --model2=model2.pb --input_node2 'input_img' --output_node2 'output_1'
