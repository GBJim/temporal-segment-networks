import subprocess
import os

output_folder = "models/ucf101/specific_models"



def deploy_prototxt(template_prefix):
    data_sources = ["flow", "rgb"]
    proto_types = ["train_val", "solver"]
    
    splits = [1,2,3] 
    for split in splits:
        for data_source in data_sources:
            for proto_type in proto_types:
                generate_prototxt(template_prefix + data_source, split, proto_type)
            

 

def generate_deploy(template_prefix, split, proto_type):
    template =  template_prefix + "_deploy_split{}.prototxt".format(split)
    output_path = os.path.join(output_folder, template_prefix)
    command = ["cd", template, output_path]
    sys.call(command)
    
def generate_prototxt(template_prefix, split, proto_type):
    template =  template_prefix + "_{}.prototxt".format(proto_type)
    
    ouput_name = os.path.basename(template_prefix) + "_{}_split{}.prototxt".format(proto_type, split)
    output_path = os.path.join(output_folder, ouput_name)
    f = open(template, 'r')
    template_content = f.read()
    f.close()
    
    output_content = template_content.replace("split1", "split{}".format(split))
    output_content = output_content.replace("split_1", "split_{}".format(split))
    w = open(output_path, 'w')
    w.write(output_content)
    w.close()
     
def generate_trainval(template_prefix, split):
    template =  template_prefix + "train_val.prototxt"
    
    ouput_name = os.path.basename(template_prefix) + "_train_val_split{}.prototxt".format(split)
    output_path = os.path.join(output_folder, ouput_name)
    f = open(template, 'r')
    template_content = f.read()
    f.close()
    
    output_content = template_content.replace("split1", "split{}".format(split))
    output_content = output_content.replace("split_1", "split_{}".format(split))
    
    w = open(output_path, 'w')
    w.write(output_content)
    w.close()
      
        
if __name__ == "__main__":
    
    print("Generating splits specific prototxt")
    template_prefix = "models/ucf101/tsn_bn_inception_" 
    deploy_prototxt(template_prefix)
    print("prototxt generated")
    
    print("Start Training Process")
    splits = [1,2,3]
    data_sources = ["flow", "rgb"]
    #data_sources = ["flow"]
    
    for split in splits:
        split = str(split)
        for data_source in data_sources:
            print("Deploy Training procees of split {} on {}".format(split, data_source))
            command = ["bash", "scripts/train_tsn_split.sh", "ucf101", data_source, split]
            #command = ["echo", "YooHoo!"]
            subprocess.call(command)

        print("Training procees of split {} on {} complete")
    print("All training procees are complete!")
    
    
    
    