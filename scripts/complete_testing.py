import subprocess
import os


FRAME_PATH = "./frames"
num_worker = str(4)


splits = ['1','2','3']
model_types_iterations = {"rgb": "4500", "flow": "20000"}

for split in splits:
    for model_type, iterations in model_types_iterations.items():
        MODEL_PATH = "models/ucf101/tsn_bn_inception_{}_deploy.prototxt".format(model_type)
        WEIGHT_PATH = "models/ucf101_split{}_tsn_{}_bn_inception_iter_{}.caffemodel".\
        format(split, model_type, iterations)
        
        SCORE_PATH = "./output/SCORE_FILE_{}_{}".format(model_type, split)
        
        cmd = ["python", "tools/eval_net.py", "ucf101", split, model_type, FRAME_PATH, MODEL_PATH, \
               WEIGHT_PATH, "--num_worker", num_worker, "--save_scores",SCORE_PATH]
        
        print(" ".join(cmd))
        subprocess.call(cmd)
        print("Testing Complete")