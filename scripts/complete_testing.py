import subprocess
import os


FRAME_PATH = "./frames"
num_worker = str(1)


splits = ['1','2','3']
model_types_iterations = {"rgb": "4500", "flow": "20000"}

for split in splits:
    for model_type, iterations in model_types_iterations.items():
        MODEL_PATH = "models/ucf101/tsn_bn_inception_{}_deploy.prototxt".format(model_type)
        WEIGHT_PATH = "models/ucf101_split{}_tsn_{}_bn_inception_iter_{}.caffemodel".\
        format(split, model_type, iterations)
        
        SCORE_PATH = "./output/SCORE_FILE_{}_{}".format(model_type, split)
        
        cmd = ["python", "tools/eval_net.py", "ucf101", split, model_type, FRAME_PATH, MODEL_PATH, \
               WEIGHT_PATH, "--num_worker", num_worker, "--save_scores",SCORE_PATH, "--gpus", "1", "2", "3"]
        
        print(" ".join(cmd))
        #subprocess.call(cmd)
        print("Testing Complete")

print("Start calculating precisions")
for split in splits:
    RGB_SCORE_PATH = "./output/SCORE_FILE_{}_{}.npz".format("rgb", split)
    FLOW_SCORE_PATH = "./output/SCORE_FILE_{}_{}.npz".format("flow", split)
    RGB_weight = str(1)
    flow_weight = str(1.5)
    cmd = ["python", "tools/eval_scores.py", RGB_SCORE_PATH, FLOW_SCORE_PATH, "--score_weights", RGB_weight, flow_weight]
    print(" ".join(cmd))
 
    #subprocess.call(cmd)
        