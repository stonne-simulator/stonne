import sys
sys.path.insert(1, 'NLP/BERT/')
sys.path.insert(1,'object_detection/ssd-mobilenets/')
sys.path.insert(1,'object_detection/ssd-resnets/')
sys.path.insert(1, 'object_detection/utils_obj_detection/')
import run_bert
import run_ssd_mobilenets
import run_ssd_resnets

BERT_WEIGHTS='NLP/BERT/best_model_state.bin'

SSD_MOBILENETS_WEIGHTS='object_detection/ssd-mobilenets/weights_tensorflow.pb'
OBJECT_DETECTION_INPUT='object_detection/tinycoco/000000037777.jpg'

# main
def run_models():
    if(len(sys.argv) != 6):
        print('Error to parse. 5 arguments were exptected.')
        print('Usage: python '+sys.argv[0]+' [name_model] [simulation_file] [tiles_path] [sparsity_ratio] [output_path]')
        exit(1)
    name_model = sys.argv[1]
    simulation_file = sys.argv[2]
    tiles_path = sys.argv[3]
    sparsity_ratio = float(sys.argv[4])
    output_path = sys.argv[5]

    if(name_model == 'bert'):
        run_bert.run_model(simulation_file, tiles_path, sparsity_ratio, output_path, BERT_WEIGHTS)
    elif(name_model == 'ssd_mobilenets'):
        run_ssd_mobilenets.run_model(simulation_file, tiles_path, sparsity_ratio, output_path, SSD_MOBILENETS_WEIGHTS, OBJECT_DETECTION_INPUT)

    elif(name_model == 'ssd_resnets'):
        print(OBJECT_DETECTION_INPUT)
        run_ssd_resnets.run_model(simulation_file, tiles_path,  sparsity_ratio, output_path, image_input=OBJECT_DETECTION_INPUT)


run_models()
